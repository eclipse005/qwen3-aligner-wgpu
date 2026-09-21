//! Model weight loading for the wgpu engine.
//!
//! Tensors are converted to f16 (bf16 via a 16-bit shift to f32, then
//! `half::f16::from_f32`) and uploaded in the layouts the kernels want:
//!
//! * fused QKV = `[q_proj | k_proj | v_proj]` rows concatenated;
//! * fused gate/up = `[gate_proj | up_proj]` rows concatenated;
//! * everything stored f16, byte-packed.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::Dtype;

/// One tensor as it sits in the file.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    /// Convert bf16/f16/f32 to f16, into a caller-provided destination.
    ///
    /// The destination form exists so a *fused* matrix (`q|k|v`, `gate|up`) can
    /// be narrowed straight into its final row-major layout: one pass, no
    /// intermediate `Vec<f16>`, and no concatenation copy afterwards.
    pub fn narrow_f16_into(&self, dst: &mut [u8]) -> Result<()> {
        let n = self.data.len() / self.dtype.size();
        anyhow::ensure!(
            dst.len() == n * 2,
            "dst is {} bytes for {n} elements, needs {}",
            dst.len(),
            n * 2
        );
        match self.dtype {
            // safetensors f16 is already little-endian, which is what the
            // kernels read, so this is the file's own bytes.
            Dtype::F16 => dst.copy_from_slice(&self.data),
            Dtype::BF16 => narrow_into(&self.data, dst, 2, |c| {
                f16::from_f32(f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            }),
            Dtype::F32 => narrow_into(&self.data, dst, 4, |c| {
                f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            }),
            other => return Err(anyhow!("unsupported dtype {other:?}")),
        }
        Ok(())
    }

    /// Convert bf16/f16/f32 to f16.
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
    /// dtype is converted element-wise.  Unlike [`Self::to_f16_vec`] this does
    /// not materialise the whole table.
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

/// mmap every safetensors shard, zero-copy.
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
///
/// `data` is `Bytes` rather than `Vec<u8>` so an f16 checkpoint can hand the
/// mapped file's own pages to the uploader: `Bytes::clone` is a refcount bump.
/// Every alternative costs whole passes over the weights — a `Vec<f16>` and then
/// a re-encoded `Vec<u8>` — before the bytes have even reached the bus.
pub struct PackedWeight {
    pub data: Bytes,
    pub rows: usize,
    pub cols: usize,
}

impl PackedWeight {
    pub fn from_f16(v: &[half::f16], rows: usize, cols: usize) -> Self {
        let mut data = Vec::with_capacity(v.len() * 2);
        for x in v {
            data.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        Self { data: data.into(), rows, cols }
    }

    /// [`RawTensor`] → packed f16, converting in one pass (and not at all when
    /// the tensor is already f16).
    ///
    /// safetensors is little-endian and little-endian f16 is exactly what the
    /// kernels read, so an f16 checkpoint hands over the mapped file's own bytes.
    ///
    /// The converting cases are the measured ones: these checkpoints are bf16,
    /// and the old route (`to_f16_vec` → `Vec<f16>` → a byte-by-byte re-encode)
    /// ran three passes over every weight on one core.  bf16 → f32 is exact (a
    /// shift), so narrowing straight to f16 with the same round-to-nearest-even
    /// is bit-identical to what that route produced.
    pub fn from_raw(t: &RawTensor, rows: usize, cols: usize) -> Result<Self> {
        let n = rows * cols;
        anyhow::ensure!(
            t.data.len() == n * t.dtype.size(),
            "{rows}x{cols} needs {} bytes, tensor has {}",
            n * t.dtype.size(),
            t.data.len()
        );
        let data: Bytes = match t.dtype {
            Dtype::F16 => t.data.clone(),
            _ => {
                let mut out = vec![0u8; n * 2];
                t.narrow_f16_into(&mut out)?;
                out.into()
            }
        };
        Ok(Self { data, rows, cols })
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
        Ok(Self { data: data.into(), rows, cols })
    }
}

/// Narrow `src` (little-endian `elem_bytes`-wide elements) to f16 in `dst`, in
/// parallel.
///
/// One pass over each of the source and destination, and the chunking is what
/// makes a multi-hundred-MiB narrowing a load-time rounding error instead of the
/// dominant cost.
fn narrow_into(
    src: &[u8],
    dst: &mut [u8],
    elem_bytes: usize,
    to_f16: impl Fn(&[u8]) -> half::f16 + Sync,
) {
    const CHUNK: usize = 1 << 16;
    dst.par_chunks_mut(CHUNK * 2)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci * CHUNK;
            for (k, slot) in chunk.chunks_exact_mut(2).enumerate() {
                let off = (base + k) * elem_bytes;
                slot.copy_from_slice(&to_f16(&src[off..off + elem_bytes]).to_bits().to_le_bytes());
            }
        });
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
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    if t.shape.len() != 2 {
        return Err(anyhow!("expected 2D weight {name}, got {:?}", t.shape));
    }
    PackedWeight::from_raw(t, t.shape[0], t.shape[1])
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
