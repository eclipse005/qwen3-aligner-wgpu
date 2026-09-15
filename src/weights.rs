//! Model weight loading for the wgpu engine.
//!
//! Deliberately independent of the main crate: `weights` / `raw_tensor` are
//! private modules there, and the brief keeps all wgpu code inside this crate.
//! The **bytes are produced by exactly the same conversion path** the CUDA
//! backend uses (bf16 -> f32 via a 16-bit shift -> `half::f16::from_f32`), so the
//! two engines consume bit-identical f16 weights.
//!
//! Tensors are uploaded in the layouts the kernels want:
//!
//! * fused QKV = `[q_proj | k_proj | v_proj]` rows concatenated (same as
//!   `load_fused_qkv_weight`);
//! * fused gate/up = `[gate_proj | up_proj]` rows concatenated;
//! * everything stored f16, byte-packed.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use memmap2::Mmap;
use safetensors::Dtype;

/// One tensor as it sits in the file.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    /// bf16/f16/f32 -> f16, matching `qwen3-asr`'s `RawTensor::to_f16_vec`.
    pub fn to_f16_vec(&self) -> Result<Vec<half::f16>> {
        Ok(match self.dtype {
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]))
                .collect(),
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::f16::from_f32)
                .collect(),
            Dtype::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    half::f16::from_f32(f32::from_bits((b as u32) << 16))
                })
                .collect(),
            other => return Err(anyhow!("unsupported dtype {other:?}")),
        })
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(match self.dtype {
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    f32::from_bits((b as u32) << 16)
                })
                .collect(),
            other => return Err(anyhow!("unsupported dtype {other:?} for to_f32_vec")),
        })
    }

    pub fn as_f16(&self) -> Result<(Vec<half::f16>, Vec<usize>)> {
        Ok((self.to_f16_vec()?, self.shape.clone()))
    }

    pub fn as_f32(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        Ok((self.to_f32_vec()?, self.shape.clone()))
    }

    /// Append row `row` of a `[*, cols]` tensor to `out` as little-endian f16 —
    /// the byte layout the decoder's `prefill` takes.
    ///
    /// An f16 tensor is a straight `memcpy` out of the mapped file; any other
    /// dtype is converted element-wise.  Use this for token lookups instead of
    /// [`Self::to_f16_vec`], which materialises the *whole* table on every call:
    /// `embed_tokens` is 155 M elements at 0.6 B and twice that at 1.7 B, i.e.
    /// ~0.3 s / ~0.6 s of allocation and conversion per transcription.
    pub fn append_f16_row_le(&self, row: usize, cols: usize, out: &mut Vec<u8>) -> Result<()> {
        anyhow::ensure!(self.shape.len() >= 1 && cols == *self.shape.last().unwrap(), "row cols {cols} != {:?}", self.shape);
        let stride = cols * 2;
        let start = row
            .checked_mul(stride)
            .filter(|s| s + stride <= self.data.len())
            .ok_or_else(|| anyhow!("row {row} of {:?} is out of range", self.shape))?;
        match self.dtype {
            Dtype::F16 => out.extend_from_slice(&self.data[start..start + stride]),
            Dtype::BF16 => {
                for c in self.data[start..start + stride].chunks_exact(2) {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    let h = half::f16::from_f32(f32::from_bits((b as u32) << 16));
                    out.extend_from_slice(&h.to_le_bytes());
                }
            }
            Dtype::F32 => {
                for c in self.data[start..].chunks_exact(4).take(cols) {
                    let h = half::f16::from_f32(f32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
                    out.extend_from_slice(&h.to_le_bytes());
                }
            }
            other => anyhow::bail!("unsupported dtype {other:?}"),
        }
        Ok(())
    }
}

/// mmap every safetensors shard, zero-copy, mirroring `weights.rs`.
pub fn load_tensors(model_dir: &Path) -> Result<HashMap<String, RawTensor>> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let idx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let wm = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("invalid index.json"))?;
        let mut shards: Vec<String> = wm
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        shards.sort();
        shards.dedup();
        let mut all = HashMap::new();
        for s in shards {
            all.extend(load_shard(&model_dir.join(&s))?);
        }
        return Ok(normalize_hf_weight_names(all));
    }
    Ok(normalize_hf_weight_names(load_shard(&model_dir.join("model.safetensors"))?))
}

fn remap_hf_weight_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("model.multi_modal_projector.linear_1") {
        return format!("thinker.audio_tower.proj1{rest}");
    }
    if let Some(rest) = name.strip_prefix("model.multi_modal_projector.linear_2") {
        return format!("thinker.audio_tower.proj2{rest}");
    }
    if let Some(rest) = name.strip_prefix("model.audio_tower.") {
        return format!("thinker.audio_tower.{rest}");
    }
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        return format!("thinker.model.{rest}");
    }
    name.to_string()
}

fn normalize_hf_weight_names(weights: HashMap<String, RawTensor>) -> HashMap<String, RawTensor> {
    let is_hf = weights.keys().any(|k| {
        k.starts_with("model.audio_tower.") || k.starts_with("model.language_model.")
    });
    if !is_hf {
        return weights;
    }
    weights
        .into_iter()
        .map(|(k, v)| (remap_hf_weight_name(&k), v))
        .collect()
}

fn load_shard(path: &Path) -> Result<HashMap<String, RawTensor>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: read-only checkpoint, never mutated while mapped.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("mmap {}", path.display()))?;
    let buf: Bytes = Bytes::from_owner(mmap);
    let st = safetensors::SafeTensors::deserialize(&buf)?;
    let base = buf.as_ptr() as usize;

    let mut out = HashMap::with_capacity(st.len());
    for (name, view) in st.iter() {
        let vd = view.data();
        let offset = vd.as_ptr() as usize - base;
        out.insert(
            name.to_string(),
            RawTensor {
                data: buf.slice(offset..offset + vd.len()),
                shape: view.shape().to_vec(),
                dtype: view.dtype(),
            },
        );
    }
    Ok(out)
}

/// A weight matrix in the final device layout: packed f16, row-major `[rows, cols]`.
pub struct PackedWeight {
    pub data: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl PackedWeight {
    pub fn from_f16(v: &[half::f16], rows: usize, cols: usize) -> Self {
        let mut data = Vec::with_capacity(v.len() * 2);
        for x in v {
            data.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        Self { data, rows, cols }
    }

    /// Concatenate row-blocks: `[a | b | c]` along the output (row) dimension.
    pub fn concat_rows(parts: &[PackedWeight], cols: usize) -> Result<Self> {
        let rows: usize = parts.iter().map(|p| p.rows).sum();
        let mut data = Vec::with_capacity(rows * cols * 2);
        for p in parts {
            if p.cols != cols {
                return Err(anyhow!(
                    "concat_rows: cols mismatch ({} vs {cols})",
                    p.cols
                ));
            }
            data.extend_from_slice(&p.data);
        }
        Ok(Self { data, rows, cols })
    }
}

/// Fetch `name`, convert to f16, and hand back both the vector and its shape.
pub fn get_f16(
    w: &HashMap<String, RawTensor>,
    name: &str,
) -> Result<(Vec<half::f16>, Vec<usize>)> {
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    Ok((t.to_f16_vec()?, t.shape.clone()))
}

pub fn get_matrix(w: &HashMap<String, RawTensor>, name: &str) -> Result<PackedWeight> {
    let (v, shape) = get_f16(w, name)?;
    if shape.len() != 2 {
        return Err(anyhow!("expected 2D weight {name}, got {shape:?}"));
    }
    Ok(PackedWeight::from_f16(&v, shape[0], shape[1]))
}

pub fn get_vector(w: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<half::f16>> {
    Ok(get_f16(w, name)?.0)
}

/// Pack a `Vec<f16>` into the u32-word layout the kernels read.
pub fn pack_words(v: &[half::f16]) -> Vec<u32> {
    assert!(v.len() % 2 == 0, "pack_words needs an even element count");
    v.chunks_exact(2)
        .map(|c| (c[0].to_bits() as u32) | ((c[1].to_bits() as u32) << 16))
        .collect()
}

/// Same, straight from little-endian bytes (avoids a round-trip through f16).
pub fn pack_words_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "pack_words_from_bytes needs 4-byte multiples");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Bytes for a `array<u32>` binding built from an f16 slice.
pub fn words_bytes(v: &[half::f16]) -> Vec<u8> {
    let words = pack_words(v);
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
