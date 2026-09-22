//! Model weight loading for the wgpu engine.
//!
//! Tensors are converted to the run's 16-bit storage format — bf16 by default,
//! f16 on `--dtype f16` (see [`crate::shaders::half`]) — and uploaded in the
//! layouts the kernels want:
//!
//! * fused QKV = `[q_proj | k_proj | v_proj]` rows concatenated;
//! * fused gate/up = `[gate_proj | up_proj]` rows concatenated;
//! * everything stored in that one format, byte-packed.
//!
//! The f16-only entry points ([`RawTensor::to_f16_vec`], [`get_f16`],
//! [`get_vector`]) belong to the CPU path: `cpu_decoder` / `cpu_tensor` are a
//! separate implementation that computes in f16, and `--dtype` does not reach
//! them.  The `h16` spellings are what the GPU path loads through.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::Dtype;

use crate::half16::H16;
use crate::shaders::{self, Half};

/// The run's storage format as a safetensors dtype — what an untouched copy of
/// a checkpoint tensor has to match ([`RawTensor::narrow_h16_into`]).
pub fn storage_dtype() -> Dtype {
    match shaders::half() {
        Half::F16 => Dtype::F16,
        Half::Bf16 => Dtype::BF16,
    }
}

/// One tensor as it sits in the file.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    /// Convert bf16/f16/f32 to the run's storage format, into a caller-provided
    /// destination.
    ///
    /// The destination form exists so a *fused* matrix (`q|k|v`, `gate|up`) can
    /// be narrowed straight into its final row-major layout: one pass, no
    /// intermediate `Vec<H16>`, and no concatenation copy afterwards.
    ///
    /// A tensor already in that format is a byte copy of the mapped file — the
    /// common case, since the checkpoint is bf16 and so is the default.
    pub fn narrow_h16_into(&self, dst: &mut [u8]) -> Result<()> {
        let half = shaders::half();
        let n = self.data.len() / self.dtype.size();
        anyhow::ensure!(
            dst.len() == n * 2,
            "dst is {} bytes for {n} elements, needs {}",
            dst.len(),
            n * 2
        );
        match (half, self.dtype) {
            // safetensors is little-endian and little-endian 16-bit floats are
            // exactly what the kernels read, so nothing needs converting.
            (Half::F16, Dtype::F16) | (Half::Bf16, Dtype::BF16) => {
                dst.copy_from_slice(&self.data);
            }
            (_, Dtype::F32) => narrow_into(&self.data, dst, 4, |c| {
                H16::from_f32_in(f32::from_le_bytes([c[0], c[1], c[2], c[3]]), half).to_bits()
            }),
            // The other 16-bit format: widen (exact — every 16-bit float is an
            // f32) and round again, which is what the reference's own
            // `.to(torch.bfloat16)` / `.to(torch.float16)` does.
            (_, Dtype::BF16) => narrow_into(&self.data, dst, 2, |c| {
                let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                H16::from_f32_in(v, half).to_bits()
            }),
            (_, Dtype::F16) => narrow_into(&self.data, dst, 2, |c| {
                let v = f16::from_le_bytes([c[0], c[1]]).to_f32();
                H16::from_f32_in(v, half).to_bits()
            }),
            (_, other) => return Err(anyhow!("unsupported dtype {other:?}")),
        }
        Ok(())
    }

    /// Convert bf16/f16/f32 to the run's storage format.
    pub fn to_h16_vec(&self) -> Result<Vec<H16>> {
        let mut bytes = vec![0u8; self.data.len() / self.dtype.size() * 2];
        self.narrow_h16_into(&mut bytes)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| H16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    /// Convert bf16/f16/f32 to f16 — the CPU path's format, see the module note.
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

    pub fn as_h16(&self) -> Result<(Vec<H16>, Vec<usize>)> {
        Ok((self.to_h16_vec()?, self.shape.clone()))
    }

    pub fn as_f16(&self) -> Result<(Vec<half::f16>, Vec<usize>)> {
        Ok((self.to_f16_vec()?, self.shape.clone()))
    }

    pub fn as_f32(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        Ok((self.to_f32_vec()?, self.shape.clone()))
    }

    /// Append row `row` of a `[*, cols]` tensor to `out` in `half`'s 16-bit
    /// format — the byte layout the decoder's `prefill` takes.
    ///
    /// A tensor already in that format is a straight `memcpy` out of the mapped
    /// file; any other dtype is converted element-wise.  Unlike
    /// [`Self::to_h16_vec`] this does not materialise the whole table, which is
    /// the point: the embedding gather reads `n_text_tokens` rows out of 152 064.
    pub fn append_row_le(
        &self,
        row: usize,
        cols: usize,
        out: &mut Vec<u8>,
        half: Half,
    ) -> Result<()> {
        anyhow::ensure!(self.shape.len() >= 1 && cols == *self.shape.last().unwrap(), "row cols {cols} != {:?}", self.shape);
        let stride = cols * 2;
        let start = row
            .checked_mul(stride)
            .filter(|s| s + stride <= self.data.len())
            .ok_or_else(|| anyhow!("row {row} of {:?} is out of range", self.shape))?;
        let elem = |c: &[u8]| -> f32 {
            match self.dtype {
                Dtype::F16 => f16::from_ne_bytes([c[0], c[1]]).to_f32(),
                Dtype::BF16 => f32::from_bits((u16::from_ne_bytes([c[0], c[1]]) as u32) << 16),
                Dtype::F32 => f32::from_ne_bytes([c[0], c[1], c[2], c[3]]),
                other => unreachable!("unsupported dtype {other:?}"),
            }
        };
        match (half, self.dtype) {
            (Half::F16, Dtype::F16) | (Half::Bf16, Dtype::BF16) => {
                out.extend_from_slice(&self.data[start..start + stride]);
            }
            (_, Dtype::F16) | (_, Dtype::BF16) | (_, Dtype::F32) => {
                let width = self.dtype.size();
                for c in self.data[start..start + stride].chunks_exact(width) {
                    out.extend_from_slice(&H16::from_f32_in(elem(c), half).to_le_bytes());
                }
            }
            (_, other) => anyhow::bail!("unsupported dtype {other:?}"),
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
    /// 16-bit elements in the run's format → packed bytes.
    pub fn from_h16(v: &[H16], rows: usize, cols: usize) -> Self {
        let mut data = Vec::with_capacity(v.len() * 2);
        for x in v {
            data.extend_from_slice(&x.to_le_bytes());
        }
        Self { data: data.into(), rows, cols }
    }

    /// [`RawTensor`] → packed weight in the run's format, converting in one pass
    /// (and not at all when the tensor is already in that format).
    ///
    /// safetensors is little-endian and little-endian 16-bit floats are exactly
    /// what the kernels read, so a checkpoint whose dtype is the run's own hands
    /// over the mapped file's own bytes — the bf16 checkpoint in the default
    /// bf16 mode, which is also the fastest case there is.
    ///
    /// The converting cases are the measured ones: the old route (`to_f16_vec` →
    /// `Vec<f16>` → a byte-by-byte re-encode) ran three passes over every weight
    /// on one core.  bf16 → f32 is exact (a shift), so narrowing straight from
    /// the file with the same round-to-nearest-even is bit-identical to what
    /// that route produced.
    pub fn from_raw(t: &RawTensor, rows: usize, cols: usize) -> Result<Self> {
        let n = rows * cols;
        anyhow::ensure!(
            t.data.len() == n * t.dtype.size(),
            "{rows}x{cols} needs {} bytes, tensor has {}",
            n * t.dtype.size(),
            t.data.len()
        );
        let data: Bytes = if t.dtype == storage_dtype() {
            t.data.clone()
        } else {
            let mut out = vec![0u8; n * 2];
            t.narrow_h16_into(&mut out)?;
            out.into()
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

/// Narrow `src` (little-endian `elem_bytes`-wide elements) to 16 bits in `dst`,
/// in parallel; `to_bits` rounds one element into the storage format.
///
/// One pass over each of the source and destination, and the chunking is what
/// makes a multi-hundred-MiB narrowing a load-time rounding error instead of the
/// dominant cost.
fn narrow_into(
    src: &[u8],
    dst: &mut [u8],
    elem_bytes: usize,
    to_bits: impl Fn(&[u8]) -> u16 + Sync,
) {
    const CHUNK: usize = 1 << 16;
    dst.par_chunks_mut(CHUNK * 2)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci * CHUNK;
            for (k, slot) in chunk.chunks_exact_mut(2).enumerate() {
                let off = (base + k) * elem_bytes;
                slot.copy_from_slice(&to_bits(&src[off..off + elem_bytes]).to_le_bytes());
            }
        });
}

/// Fetch `name`, convert to the run's storage format, and hand back both the
/// vector and its shape.
pub fn get_h16(
    w: &HashMap<String, RawTensor>,
    name: &str,
) -> Result<(Vec<H16>, Vec<usize>)> {
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    Ok((t.to_h16_vec()?, t.shape.clone()))
}

/// Fetch `name`, convert to f16, and hand back both the vector and its shape.
///
/// f16 is the CPU path's format — see the module note.
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

pub fn get_h16_vector(w: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<H16>> {
    Ok(get_h16(w, name)?.0)
}

pub fn get_vector(w: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<half::f16>> {
    Ok(get_f16(w, name)?.0)
}

/// Pack a slice of 16-bit elements into the u32-word layout the kernels read.
///
/// Nothing here depends on the format: two 16-bit elements per `u32`, element 0
/// in the low half.
pub fn pack_words(v: &[H16]) -> Vec<u32> {
    assert!(v.len() % 2 == 0, "pack_words needs an even element count");
    v.chunks_exact(2)
        .map(|c| (c[0].to_bits() as u32) | ((c[1].to_bits() as u32) << 16))
        .collect()
}

/// Same, straight from little-endian bytes (avoids a round-trip through `H16`).
pub fn pack_words_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "pack_words_from_bytes needs 4-byte multiples");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Bytes for a `array<u32>` binding built from a slice of 16-bit elements.
pub fn words_bytes(v: &[H16]) -> Vec<u8> {
    let words = pack_words(v);
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
