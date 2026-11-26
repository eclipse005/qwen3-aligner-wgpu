//! The optional **Local Head**: coarse boundaries refined to 10 ms.
//!
//! The Qwen head can only place an endpoint on its 80 ms class grid.  The Local
//! Head from `17slever17/Qwen3-ForcedAligner-Ultra` refines each endpoint inside
//! a bounded `±60 ms` neighbourhood, from two inputs that are already on this
//! side of the fence:
//!
//! * **context** — the final-norm hidden rows at the word's two `<timestamp>`
//!   slots, the word's mean, and its neighbours' means (`[4, 1024]`), with a
//!   mask that drops the missing neighbour at the utterance edges;
//! * **acoustics** — a `[41, 91]` window centred on the *coarse* endpoint
//!   (`±200 ms`, 10 ms hop, 25 ms frames): 80 log-mels, log-RMS, its delta,
//!   spectral flux, and 8 spectral-distance channels.
//!
//! The network is 405k parameters: two 5-tap convolutions over the acoustic
//! window, a role/endpoint-conditioned query attended over the four context
//! rows and then over the 41 frames, a per-frame scoring head over the 13
//! candidate offsets (`±60 ms` in 10 ms steps), a first-difference term, and a
//! Gaussian prior (`σ = 35 ms`, ×2) that alone reproduces the coarse bucket.
//! The output is the **posterior mean** of the offsets at temperature 1.5.
//!
//! This module is a transcription of `qwen_fa_ultra` (`runtime.py`,
//! `features.py`, `head.py`) that ships with the weights, and is f32 throughout
//! like that reference.  The order matters and is kept:
//!
//! 1. mask the timestamp classes that fall past the audio's end, then argmax —
//!    this masked stream is what the head refines (the base model never masks);
//! 2. `fix_timestamps` on that stream;
//! 3. features are centred on the **fixed** values, not on the corrections;
//! 4. each endpoint is corrected independently;
//! 5. endpoints closer than 10 ms are projected around their midpoint, overlaps
//!    are trimmed trusting the **next** word's start, and everything is bounded
//!    to the audio duration.
//!
//! The package is validated as strictly as the reference validates it: contract
//! literals, tensor shapes, the 91-value normalizer and the `qwen_prior` buffer
//! all have to match, because a head that silently accepted a different contract
//! would produce plausible-looking but meaningless offsets.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use safetensors::Dtype;
use serde::Deserialize;

use crate::postprocess::{fix_timestamps, AlignItem};

/// Where a checkpoint keeps its Local Head, when it ships one.
pub const DIR: &str = "local_head";

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_MS: usize = 25;
pub const HOP_MS: usize = 10;
pub const CONTEXT_MS: usize = 200;
pub const FRAME_COUNT: usize = 41;
pub const N_FFT: usize = 512;
pub const N_MELS: usize = 80;
pub const SPECTRAL_BANDS: usize = 4;
pub const FEATURE_DIM: usize = N_MELS + 3 + 2 * SPECTRAL_BANDS;
pub const LOG_EPSILON: f32 = 1e-6;

pub const SEGMENT_MS: f64 = 80.0;
pub const MIN_WORD_DURATION_MS: f64 = 10.0;
pub const DECODER_TEMPERATURE: f32 = 1.5;
pub const OFFSETS_MS: [f32; 13] = [
    -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
];

const SCHEMA: &str = "qwen-fa-ultra.local-head.v1";
const ARCHITECTURE: &str = "IdentityConditionedBoundaryHead";
const CONTEXT_ROLES: [&str; 4] = ["timestamp slot", "current avg", "prev avg", "next avg"];
const PRIOR_SIGMA_MS: f64 = 35.0;
const INFERENCE_PRIOR_SCALE: f64 = 2.0;

/// `[4]` per endpoint: which of the context rows carry information.
pub type ContextMask = [f32; 4];

/// The optional refinement stage of an [`crate::align_inference::Aligner`].
pub struct LocalHead {
    weights: Weights,
    /// `[91]` train-stat normalizer, applied after the feature window.
    mean: Vec<f32>,
    std: Vec<f32>,
    /// 512-point FFT for the 41 frames of one endpoint.
    fft: Arc<dyn Fft<f64>>,
    hann: Vec<f32>,
    /// `[80][257]` mel bank, normalized per row like the reference.
    bank: Vec<f32>,
    /// Per-mel active `[start_bin, end_bin]`; outside this range the bank is
    /// exactly zero, so summing only this slice is bit-identical to the full
    /// 257-bin loop in Python.
    mel_ranges: Vec<(usize, usize)>,
    /// In-place scratch length required by the planned FFT.
    fft_scratch_len: usize,
}

impl std::fmt::Debug for LocalHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalHead")
            .field("parameters", &self.weights.numel())
            .finish()
    }
}

impl LocalHead {
    /// `<model>/local_head`, the location the reference resolves.
    pub fn package_dir(model_dir: &Path) -> PathBuf {
        model_dir.join(DIR)
    }

    /// Load the package next to a checkpoint; `None` when the checkpoint ships
    /// none, an error when it ships one that does not validate.
    pub fn load(model_dir: &Path) -> Result<Option<Self>> {
        let dir = Self::package_dir(model_dir);
        if !dir.is_dir() {
            return Ok(None);
        }
        let config: HeadFile = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .with_context(|| format!("read {}", dir.join("config.json").display()))?,
        )
        .context("parse local_head/config.json")?;
        config.validate()?;

        let normalizer: NormalizerFile = serde_json::from_str(
            &std::fs::read_to_string(dir.join("normalizer.json"))
                .with_context(|| format!("read {}", dir.join("normalizer.json").display()))?,
        )
        .context("parse local_head/normalizer.json")?;
        ensure!(
            normalizer.mean.len() == FEATURE_DIM && normalizer.std.len() == FEATURE_DIM,
            "normalizer must hold {FEATURE_DIM} mean/std values, got {}/{}",
            normalizer.mean.len(),
            normalizer.std.len()
        );
        ensure!(
            normalizer.mean.iter().all(|v| v.is_finite())
                && normalizer.std.iter().all(|v| v.is_finite() && *v > 0.0),
            "normalizer must be finite with positive std"
        );

        let weights = Weights::load(&dir.join("model.safetensors"), &config.model_config)?;
        let expected = canonical_prior();
        ensure!(
            weights
                .prior
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() <= 1e-5),
            "local_head qwen_prior is not the canonical production prior \
             (σ={PRIOR_SIGMA_MS} ms ×{INFERENCE_PRIOR_SCALE})"
        );

        let mut planner = rustfft::FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let fft_scratch_len = fft.get_inplace_scratch_len();
        let bank = mel_filterbank();
        let mel_ranges = mel_active_ranges(&bank);
        Ok(Some(Self {
            weights,
            mean: normalizer.mean,
            std: normalizer.std,
            fft,
            hann: hann_window(FRAME_MS * SAMPLE_RATE / 1000),
            bank,
            mel_ranges,
            fft_scratch_len,
        }))
    }

    pub fn num_parameters(&self) -> usize {
        self.weights.numel()
    }

    pub fn describe(&self) -> String {
        format!("{} parameters", self.num_parameters())
    }

    /// Refine one utterance.
    ///
    /// * `samples` — mono 16 kHz audio, the same array the towers saw;
    /// * `words` — the split transcript, one entry per aligned word;
    /// * `logits` — `[2 * words, n_labels]`, the timestamp head's pre-argmax
    ///   output (before the coarse repair);
    /// * `hidden` — `[2 * words, hidden_size]`, the final-norm rows at the
    ///   timestamp slots, in slot order.
    pub fn refine(
        &self,
        samples: &[f32],
        words: &[String],
        logits: &[f32],
        n_labels: usize,
        hidden: &[f32],
        hidden_size: usize,
    ) -> Result<Vec<AlignItem>> {
        let word_count = words.len();
        let slots = word_count * 2;
        ensure!(
            logits.len() == slots * n_labels,
            "timestamp logits are {} values, expected {slots}x{n_labels}",
            logits.len()
        );
        ensure!(
            hidden.len() == slots * hidden_size,
            "hidden rows are {} values, expected {slots}x{hidden_size}",
            hidden.len()
        );
        ensure!(
            hidden_size == self.weights.context_in,
            "hidden size {hidden_size} != local head context input {}",
            self.weights.context_in
        );
        ensure!(!samples.is_empty(), "local head needs non-empty audio");
        ensure!(n_labels > 0, "timestamp head has no classes");

        if word_count == 0 {
            return Ok(Vec::new());
        }

        let duration_ms = samples.len() as f64 * 1000.0 / SAMPLE_RATE as f64;
        let valid_classes = valid_class_count(duration_ms, n_labels);

        // 1. duration-masked argmax, then the official repair
        let raw_ms: Vec<i64> = (0..slots)
            .map(|slot| {
                let row = &logits[slot * n_labels..(slot + 1) * n_labels];
                argmax_ms(&row[..valid_classes])
            })
            .collect();
        let fixed_ms = fix_timestamps(&raw_ms);

        // 2. the 91-channel window centred on each fixed endpoint
        let features = self.features(samples, &fixed_ms);

        // 3. per-endpoint offsets (pre-projects slot rows & word means once,
        //    then runs per-endpoint GEMM + attention with per-thread scratch)
        let offsets = self.refine_offsets(&features, hidden, hidden_size, word_count);

        // 5. repair the corrected intervals, in the reference's order
        let mut intervals: Vec<[f64; 2]> = (0..word_count)
            .map(|i| {
                let fixed_start = fixed_ms[i * 2] as f64;
                let fixed_end = fixed_ms[i * 2 + 1] as f64;
                let mut start = fixed_start + offsets[i * 2] as f64;
                let mut end = fixed_end + offsets[i * 2 + 1] as f64;
                if !start.is_finite() || !end.is_finite() {
                    start = fixed_start;
                    end = fixed_end;
                } else if end - start < MIN_WORD_DURATION_MS {
                    let midpoint = (start + end) / 2.0;
                    if midpoint < MIN_WORD_DURATION_MS / 2.0 {
                        start = 0.0;
                        end = MIN_WORD_DURATION_MS;
                    } else {
                        start = midpoint - MIN_WORD_DURATION_MS / 2.0;
                        end = midpoint + MIN_WORD_DURATION_MS / 2.0;
                    }
                } else if start < 0.0 {
                    end -= start;
                    start = 0.0;
                }
                if !start.is_finite()
                    || !end.is_finite()
                    || start < 0.0
                    || end < 0.0
                    || end - start < MIN_WORD_DURATION_MS
                {
                    start = 0.0;
                    end = MIN_WORD_DURATION_MS;
                }
                [start, end]
            })
            .collect();

        remove_neighbor_overlaps(&mut intervals);
        bound_to_duration(&mut intervals, duration_ms);

        Ok(words
            .iter()
            .enumerate()
            .map(|(i, w)| AlignItem {
                text: w.clone(),
                start_time: intervals[i][0] / 1000.0,
                end_time: intervals[i][1] / 1000.0,
            })
            .collect())
    }

    /// Project the unique slot rows and per-word means once via GEMM, then
    /// evaluate all endpoints in parallel with per-thread scratch reuse.
    fn refine_offsets(
        &self,
        features: &[f32],
        hidden: &[f32],
        hidden_size: usize,
        word_count: usize,
    ) -> Vec<f32> {
        let slots = word_count * 2;
        if slots == 0 {
            return Vec::new();
        }
        let w = &self.weights;
        let dim = w.model_dim;

        // Compute per-word mean rows in the exact f32 order of `context_rows`.
        let mut word_means = vec![0.0f32; word_count * hidden_size];
        word_means
            .par_chunks_mut(hidden_size)
            .enumerate()
            .for_each(|(word, dst)| {
                let r0 = &hidden[(word * 2) * hidden_size..(word * 2 + 1) * hidden_size];
                let r1 = &hidden[(word * 2 + 1) * hidden_size..(word * 2 + 2) * hidden_size];
                for ((o, &a), &b) in dst.iter_mut().zip(r0).zip(r1) {
                    *o = (a + b) * 0.5;
                }
            });

        // Project each unique hidden row and word mean once instead of 6x.
        let mut slot_proj = vec![0.0f32; slots * dim];
        let mut mean_proj = vec![0.0f32; word_count * dim];
        gemm_row_major_par(&mut slot_proj, hidden, &w.ctx_w, slots, dim, hidden_size, 0.0);
        gemm_row_major_par(
            &mut mean_proj,
            &word_means,
            &w.ctx_w,
            word_count,
            dim,
            hidden_size,
            0.0,
        );

        let mut offsets = vec![0.0f32; slots];
        const CHUNK: usize = 8;
        offsets
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each_init(
                || EndpointScratch::new(dim),
                |scratch, (chunk_index, out_chunk)| {
                    let start = chunk_index * CHUNK;
                    for (j, out) in out_chunk.iter_mut().enumerate() {
                        let endpoint = start + j;
                        let word = endpoint / 2;
                        let s_row = &slot_proj[endpoint * dim..(endpoint + 1) * dim];
                        let cur_row = &mean_proj[word * dim..(word + 1) * dim];
                        let prev_row = if word > 0 {
                            Some(&mean_proj[(word - 1) * dim..word * dim])
                        } else {
                            None
                        };
                        let next_row = if word + 1 < word_count {
                            Some(&mean_proj[(word + 1) * dim..(word + 2) * dim])
                        } else {
                            None
                        };

                        for d in 0..dim {
                            let b = w.ctx_b[d];
                            scratch.tokens[d] = s_row[d] + (b + w.role[d]);
                            scratch.tokens[dim + d] = cur_row[d] + (b + w.role[dim + d]);
                            scratch.tokens[2 * dim + d] =
                                prev_row.map_or(0.0, |r| r[d]) + (b + w.role[2 * dim + d]);
                            scratch.tokens[3 * dim + d] =
                                next_row.map_or(0.0, |r| r[d]) + (b + w.role[3 * dim + d]);
                        }
                        let mask: ContextMask = [
                            1.0,
                            1.0,
                            if word > 0 { 1.0 } else { 0.0 },
                            if word + 1 < word_count { 1.0 } else { 0.0 },
                        ];
                        let feat = &features[endpoint * FRAME_COUNT * FEATURE_DIM
                            ..(endpoint + 1) * FRAME_COUNT * FEATURE_DIM];
                        *out = self.endpoint_offset_from_tokens(
                            feat,
                            &mask,
                            endpoint % 2,
                            scratch,
                        );
                    }
                },
            );
        offsets
    }

    /// `_context_for_sample`: `[slot, current avg, prev avg, next avg]` per
    /// endpoint, with the missing neighbours at the edges as zeros but flagged
    /// in the mask.
    #[cfg(test)]
    fn context_rows(
        &self,
        hidden: &[f32],
        hidden_size: usize,
        word_count: usize,
    ) -> (Vec<f32>, Vec<ContextMask>) {
        let row = |i: usize| &hidden[i * hidden_size..(i + 1) * hidden_size];
        let mean = |a: usize, b: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; hidden_size];
            for i in a..b {
                for (o, v) in out.iter_mut().zip(row(i)) {
                    *o += v;
                }
            }
            let n = (b - a) as f32;
            out.iter_mut().for_each(|v| *v /= n);
            out
        };

        let mut context = vec![0.0f32; word_count * 2 * 4 * hidden_size];
        let mut masks = vec![[0.0f32; 4]; word_count * 2];
        for w in 0..word_count {
            let current = mean(w * 2, w * 2 + 2);
            let previous = if w > 0 {
                mean(w * 2 - 2, w * 2)
            } else {
                vec![0.0; hidden_size]
            };
            let next = if w + 1 < word_count {
                mean(w * 2 + 2, w * 2 + 4)
            } else {
                vec![0.0; hidden_size]
            };
            for endpoint in 0..2 {
                let slot = w * 2 + endpoint;
                let base = slot * 4 * hidden_size;
                let roles = [row(slot), &current[..], &previous[..], &next[..]];
                for (role, source) in roles.into_iter().enumerate() {
                    let dst =
                        &mut context[base + role * hidden_size..base + (role + 1) * hidden_size];
                    dst.copy_from_slice(source);
                }
                masks[slot] = [
                    1.0,
                    1.0,
                    if w > 0 { 1.0 } else { 0.0 },
                    if w + 1 < word_count { 1.0 } else { 0.0 },
                ];
            }
        }
        (context, masks)
    }

    /// `extract_fixed_center_features`: `[N][41][91]`, normalized.
    fn features(&self, samples: &[f32], centers_ms: &[i64]) -> Vec<f32> {
        let mut out = vec![0.0f32; centers_ms.len() * FRAME_COUNT * FEATURE_DIM];
        if centers_ms.is_empty() {
            return out;
        }
        // The 80 ms coarse grid is always a multiple of the 10 ms hop; the
        // global path only applies when *every* center is on that grid.
        let hop_ms = HOP_MS as i64;
        if centers_ms.iter().any(|&c| c % hop_ms != 0) {
            out.par_chunks_mut(FRAME_COUNT * FEATURE_DIM)
                .zip(centers_ms.par_iter())
                .for_each(|(dst, &center_ms)| self.features_one(samples, center_ms, dst));
            return out;
        }

        let half_ctx = (CONTEXT_MS / HOP_MS) as i64; // 20
        let hop = (HOP_MS * SAMPLE_RATE / 1000) as i64; // 160 samples
        let frame_len = FRAME_MS * SAMPLE_RATE / 1000; // 400 samples
        let bins = N_FFT / 2 + 1; // 257

        let mut min_frame = i64::MAX;
        let mut max_frame = i64::MIN;
        for &center_ms in centers_ms {
            let center_frame = center_ms / hop_ms;
            min_frame = min_frame.min(center_frame - half_ctx);
            max_frame = max_frame.max(center_frame + (FRAME_COUNT as i64 - 1 - half_ctx));
        }
        let total_frames = (max_frame - min_frame + 1) as usize;
        // Guard against pathological sparse inputs (a few endpoints very far
        // apart): a per-endpoint pass is cheaper than a giant global span.
        if total_frames > centers_ms.len() * FRAME_COUNT {
            out.par_chunks_mut(FRAME_COUNT * FEATURE_DIM)
                .zip(centers_ms.par_iter())
                .for_each(|(dst, &center_ms)| self.features_one(samples, center_ms, dst));
            return out;
        }

        // Mark frames that at least one endpoint actually uses; long silence
        // gaps between sparse clusters are then skipped.
        let mut needed = vec![false; total_frames];
        for &center_ms in centers_ms {
            let center_frame = center_ms / hop_ms;
            let first = center_frame - half_ctx - min_frame;
            let last = first + FRAME_COUNT as i64 - 1;
            for gi in first as usize..=last as usize {
                needed[gi] = true;
            }
        }

        let mut log_rms = vec![0.0f32; total_frames];
        let mut magnitude = vec![0.0f32; total_frames * bins];
        let mut log_mel = vec![0.0f32; total_frames * N_MELS];
        let mut normalised = vec![0.0f32; total_frames * bins];

        // One FFT per needed global frame, parallel over chunks of frames.
        // FFT buffer + scratch are allocated once per rayon worker thread.
        const FRAME_CHUNK: usize = 64;
        log_rms
            .par_chunks_mut(FRAME_CHUNK)
            .zip(magnitude.par_chunks_mut(FRAME_CHUNK * bins))
            .zip(log_mel.par_chunks_mut(FRAME_CHUNK * N_MELS))
            .zip(normalised.par_chunks_mut(FRAME_CHUNK * bins))
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![Complex::new(0.0f64, 0.0); N_FFT],
                        vec![Complex::new(0.0f64, 0.0); self.fft_scratch_len],
                    )
                },
                |(buffer, scratch), (chunk_idx, (((rms_chunk, mag_chunk), mel_chunk), norm_chunk))| {
                    let base = chunk_idx * FRAME_CHUNK;
                    let mut frame = [0.0f32; 400];
                    let mut power = [0.0f32; N_FFT / 2 + 1];
                    for local_gi in 0..rms_chunk.len() {
                        let gi = base + local_gi;
                        if !needed[gi] {
                            continue;
                        }
                        let g = min_frame + gi as i64;
                        let start = g * hop - (frame_len / 2) as i64;
                        let mut acc = 0.0f32;
                        if start >= 0 && (start as usize) + frame_len <= samples.len() {
                            let src = &samples[start as usize..start as usize + frame_len];
                            for i in 0..frame_len {
                                let v = src[i] * self.hann[i];
                                frame[i] = v;
                                acc += v * v;
                            }
                        } else {
                            frame.fill(0.0);
                            for i in 0..frame_len {
                                let index = start + i as i64;
                                if index >= 0 && (index as usize) < samples.len() {
                                    let v = samples[index as usize] * self.hann[i];
                                    frame[i] = v;
                                    acc += v * v;
                                }
                            }
                        }
                        rms_chunk[local_gi] = (acc / frame_len as f32 + LOG_EPSILON)
                            .sqrt()
                            .max(LOG_EPSILON)
                            .ln();

                        for i in 0..frame_len {
                            buffer[i] = Complex::new(frame[i] as f64, 0.0);
                        }
                        for b in &mut buffer[frame_len..N_FFT] {
                            *b = Complex::new(0.0, 0.0);
                        }
                        self.fft.process_with_scratch(buffer, scratch);
                        let mag = &mut mag_chunk[local_gi * bins..(local_gi + 1) * bins];
                        for b in 0..bins {
                            let m = buffer[b].norm() as f32;
                            mag[b] = m;
                            power[b] = m * m;
                        }

                        // Sparse mel accumulation: bit-identical, ~25x fewer multiplies.
                        for m in 0..N_MELS {
                            let (start_bin, end_bin) = self.mel_ranges[m];
                            let bank = &self.bank[m * bins + start_bin..m * bins + end_bin];
                            let mut acc = 0.0f32;
                            for (b, &weight) in bank.iter().enumerate() {
                                acc += weight * power[start_bin + b];
                            }
                            mel_chunk[local_gi * N_MELS + m] = acc.max(LOG_EPSILON).ln();
                        }

                        let row = &mut norm_chunk[local_gi * bins..(local_gi + 1) * bins];
                        let mut mean = 0.0f32;
                        for (b, v) in row.iter_mut().enumerate() {
                            let value = mag[b].max(LOG_EPSILON).ln();
                            *v = value;
                            mean += value;
                        }
                        mean /= bins as f32;
                        for v in row.iter_mut() {
                            *v -= mean;
                        }
                    }
                },
            );

        // Adjacent-frame features, computed once for each needed pair.
        let edges = band_edges(bins);
        let mut adj_flux = vec![0.0f32; total_frames];
        let mut adj_dist = vec![0.0f32; total_frames * SPECTRAL_BANDS];
        adj_flux
            .par_chunks_mut(FRAME_CHUNK)
            .zip(adj_dist.par_chunks_mut(FRAME_CHUNK * SPECTRAL_BANDS))
            .enumerate()
            .for_each(|(chunk_idx, (fchunk, dchunk))| {
                let first = chunk_idx * FRAME_CHUNK;
                for offset in 0..fchunk.len() {
                    let gi = first + offset;
                    if gi == 0 || !needed[gi] || !needed[gi - 1] {
                        continue;
                    }
                    let cur = &magnitude[gi * bins..(gi + 1) * bins];
                    let prev = &magnitude[(gi - 1) * bins..gi * bins];
                    let mut positive = 0.0f32;
                    let mut total = 0.0f32;
                    for b in 0..bins {
                        positive += (cur[b] - prev[b]).max(0.0);
                        total += prev[b];
                    }
                    fchunk[offset] = positive / (total + LOG_EPSILON);
                    for band in 0..SPECTRAL_BANDS {
                        let left = edges[band];
                        let right = edges[band + 1].max(left + 1);
                        let a = &normalised[(gi - 1) * bins + left..(gi - 1) * bins + right];
                        let b = &normalised[gi * bins + left..gi * bins + right];
                        let mut acc = 0.0f32;
                        for i in 0..a.len() {
                            acc += (b[i] - a[i]).abs();
                        }
                        dchunk[offset * SPECTRAL_BANDS + band] = acc / a.len() as f32;
                    }
                }
            });

        // Assemble each endpoint's 41-frame window from the global arrays.
        // par_chunks_mut keeps each output region owned by one rayon worker.
        out.par_chunks_mut(FRAME_COUNT * FEATURE_DIM)
            .zip(centers_ms.par_iter())
            .for_each(|(dst, &center_ms)| {
                let center_frame = center_ms / hop_ms;
                let base_gi = (center_frame - half_ctx - min_frame) as usize;
                for f in 0..FRAME_COUNT {
                    let gi = base_gi + f;
                    let row = &mut dst[f * FEATURE_DIM..(f + 1) * FEATURE_DIM];
                    row[..N_MELS].copy_from_slice(&log_mel[gi * N_MELS..(gi + 1) * N_MELS]);
                    row[N_MELS] = log_rms[gi];
                    row[N_MELS + 1] = if f == 0 {
                        0.0
                    } else {
                        log_rms[gi] - log_rms[gi - 1]
                    };
                    row[N_MELS + 2] = if f == 0 { 0.0 } else { adj_flux[gi] };
                    for band in 0..SPECTRAL_BANDS {
                        row[N_MELS + 3 + band] = if f == 0 {
                            0.0
                        } else {
                            adj_dist[gi * SPECTRAL_BANDS + band]
                        };
                        row[N_MELS + 3 + SPECTRAL_BANDS + band] = if f + 1 == FRAME_COUNT {
                            0.0
                        } else {
                            adj_dist[(gi + 1) * SPECTRAL_BANDS + band]
                        };
                    }
                    for ((value, mean), std) in row.iter_mut().zip(&self.mean).zip(&self.std) {
                        *value = (*value - *mean) / *std;
                    }
                }
            });
        out
    }

    /// Fallback path: one endpoint's local 41-frame window, used whenever a
    /// center is off the 10 ms grid or the input is too sparse for the global
    /// pass.  It computes the same per-frame primitives in place.
    fn features_one(&self, samples: &[f32], center_ms: i64, dst: &mut [f32]) {
        let frame_len = FRAME_MS * SAMPLE_RATE / 1000; // 400
        let hop = (HOP_MS * SAMPLE_RATE / 1000) as i64; // 160
        let bins = N_FFT / 2 + 1; // 257
        let half_ctx = (CONTEXT_MS / HOP_MS) as i64; // 20
        let center_sample = center_ms * (SAMPLE_RATE as i64) / 1000;

        let mut log_rms = [0.0f32; FRAME_COUNT];
        let mut magnitude = vec![0.0f32; FRAME_COUNT * bins];
        let mut log_mel = vec![0.0f32; FRAME_COUNT * N_MELS];
        let mut normalised = vec![0.0f32; FRAME_COUNT * bins];
        let mut buffer = vec![Complex::new(0.0f64, 0.0); N_FFT];
        let mut scratch = vec![Complex::new(0.0f64, 0.0); self.fft_scratch_len];
        for f in 0..FRAME_COUNT {
            let start = center_sample + (f as i64 - half_ctx) * hop - (frame_len / 2) as i64;
            let mut frame = [0.0f32; 400];
            for i in 0..frame_len {
                let index = start + i as i64;
                if index >= 0 && (index as usize) < samples.len() {
                    frame[i] = samples[index as usize] * self.hann[i];
                }
            }
            let mut acc = 0.0f32;
            for &v in &frame {
                acc += v * v;
            }
            log_rms[f] = (acc / frame_len as f32 + LOG_EPSILON)
                .sqrt()
                .max(LOG_EPSILON)
                .ln();

            buffer.fill(Complex::new(0.0, 0.0));
            for i in 0..frame_len {
                buffer[i].re = frame[i] as f64;
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            let mag = &mut magnitude[f * bins..(f + 1) * bins];
            for b in 0..bins {
                mag[b] = buffer[b].norm() as f32;
            }
            for m in 0..N_MELS {
                let (start_bin, end_bin) = self.mel_ranges[m];
                let bank = &self.bank[m * bins + start_bin..m * bins + end_bin];
                let mut acc = 0.0f32;
                for (b, &weight) in bank.iter().enumerate() {
                    let power = mag[start_bin + b] * mag[start_bin + b];
                    acc += weight * power;
                }
                log_mel[f * N_MELS + m] = acc.max(LOG_EPSILON).ln();
            }
            let row = &mut normalised[f * bins..(f + 1) * bins];
            let mut mean = 0.0f32;
            for (b, v) in row.iter_mut().enumerate() {
                let value = mag[b].max(LOG_EPSILON).ln();
                *v = value;
                mean += value;
            }
            mean /= bins as f32;
            for v in row.iter_mut() {
                *v -= mean;
            }
        }

        let edges = band_edges(bins);
        for f in 0..FRAME_COUNT {
            let row = &mut dst[f * FEATURE_DIM..(f + 1) * FEATURE_DIM];
            row[..N_MELS].copy_from_slice(&log_mel[f * N_MELS..(f + 1) * N_MELS]);
            row[N_MELS] = log_rms[f];
            row[N_MELS + 1] = if f == 0 {
                0.0
            } else {
                log_rms[f] - log_rms[f - 1]
            };
            if f == 0 {
                row[N_MELS + 2] = 0.0;
            } else {
                let cur = &magnitude[f * bins..(f + 1) * bins];
                let prev = &magnitude[(f - 1) * bins..f * bins];
                let mut positive = 0.0f32;
                let mut total = 0.0f32;
                for b in 0..bins {
                    positive += (cur[b] - prev[b]).max(0.0);
                    total += prev[b];
                }
                row[N_MELS + 2] = positive / (total + LOG_EPSILON);
            }
            for band in 0..SPECTRAL_BANDS {
                let left = edges[band];
                let right = edges[band + 1].max(left + 1);
                let cur = &normalised[f * bins + left..f * bins + right];
                let near_l = if f == 0 {
                    cur
                } else {
                    &normalised[(f - 1) * bins + left..(f - 1) * bins + right]
                };
                let near_r = if f + 1 == FRAME_COUNT {
                    cur
                } else {
                    &normalised[(f + 1) * bins + left..(f + 1) * bins + right]
                };
                let mut acc_l = 0.0f32;
                let mut acc_r = 0.0f32;
                for i in 0..cur.len() {
                    acc_l += (cur[i] - near_l[i]).abs();
                    acc_r += (cur[i] - near_r[i]).abs();
                }
                let width = cur.len() as f32;
                row[N_MELS + 3 + band] = acc_l / width;
                row[N_MELS + 3 + SPECTRAL_BANDS + band] = acc_r / width;
            }
            for ((value, mean), std) in row.iter_mut().zip(&self.mean).zip(&self.std) {
                *value = (*value - *mean) / *std;
            }
        }
    }

    /// One endpoint through `IdentityConditionedBoundaryHead`; the decoded
    /// offset in milliseconds.
    #[cfg(test)]
    fn endpoint_offset(&self, feat: &[f32], ctx: &[f32], mask: &ContextMask, is_end: usize) -> f32 {
        let w = &self.weights;
        let dim = w.model_dim;
        let mut scratch = EndpointScratch::new(dim);
        gemm_row_major(
            &mut scratch.tokens,
            ctx,
            &w.ctx_w,
            4,
            dim,
            w.context_in,
            0.0,
        );
        for r in 0..4 {
            for d in 0..dim {
                let i = r * dim + d;
                scratch.tokens[i] += w.ctx_b[d] + w.role[r * dim + d];
            }
        }
        self.endpoint_offset_from_tokens(feat, mask, is_end, &mut scratch)
    }

    /// Fast single-endpoint path assuming `scratch.tokens` (`[4, dim]`) is
    /// already populated with projected context tokens + bias + role embedding.
    fn endpoint_offset_from_tokens(
        &self,
        feat: &[f32],
        mask: &ContextMask,
        is_end: usize,
        scratch: &mut EndpointScratch,
    ) -> f32 {
        let w = &self.weights;
        let dim = w.model_dim;
        let frames = FRAME_COUNT;

        // ---- conv1: [41][91] im2col -> [41, 128] -> GELU ----
        let im1 = &mut scratch.im1;
        for t in 0..frames {
            let row = &mut im1[t * 5 * FEATURE_DIM..(t + 1) * 5 * FEATURE_DIM];
            if t >= 2 && t + 2 < frames {
                row.copy_from_slice(&feat[(t - 2) * FEATURE_DIM..(t + 3) * FEATURE_DIM]);
            } else {
                row.fill(0.0);
                for k in 0..5 {
                    let tt = t as isize + k as isize - 2;
                    if (0..frames as isize).contains(&tt) {
                        let src =
                            &feat[tt as usize * FEATURE_DIM..(tt as usize + 1) * FEATURE_DIM];
                        row[k * FEATURE_DIM..(k + 1) * FEATURE_DIM].copy_from_slice(src);
                    }
                }
            }
        }
        gemm_row_major(
            &mut scratch.hidden_a,
            im1,
            &w.conv1_w,
            frames,
            dim,
            5 * FEATURE_DIM,
            0.0,
        );
        for t in 0..frames {
            let row = &mut scratch.hidden_a[t * dim..(t + 1) * dim];
            for (v, &b) in row.iter_mut().zip(&w.conv1_b) {
                *v = gelu(*v + b);
            }
        }

        // ---- conv2: [41][128] im2col -> [41, 128] -> GELU ----
        let im2 = &mut scratch.im2;
        for t in 0..frames {
            let row = &mut im2[t * 5 * dim..(t + 1) * 5 * dim];
            if t >= 2 && t + 2 < frames {
                row.copy_from_slice(&scratch.hidden_a[(t - 2) * dim..(t + 3) * dim]);
            } else {
                row.fill(0.0);
                for k in 0..5 {
                    let tt = t as isize + k as isize - 2;
                    if (0..frames as isize).contains(&tt) {
                        let src = &scratch.hidden_a[tt as usize * dim..(tt as usize + 1) * dim];
                        row[k * dim..(k + 1) * dim].copy_from_slice(src);
                    }
                }
            }
        }
        gemm_row_major(
            &mut scratch.audio,
            im2,
            &w.conv2_w,
            frames,
            dim,
            5 * dim,
            0.0,
        );
        for t in 0..frames {
            let row = &mut scratch.audio[t * dim..(t + 1) * dim];
            for (v, &b) in row.iter_mut().zip(&w.conv2_b) {
                *v = gelu(*v + b);
            }
        }

        // ---- context attention (4 masked keys) ----
        let tokens = &scratch.tokens[..4 * dim];
        let base_query = &mut scratch.base_query[..dim];
        for d in 0..dim {
            base_query[d] = tokens[d] + w.endpoint[is_end * dim + d];
        }
        matvec_bias(&mut scratch.cq, base_query, &w.cq, &w.cbq, dim, dim);
        gemm_row_major(&mut scratch.ck, tokens, &w.ck, 4, dim, dim, 0.0);
        gemm_row_major(&mut scratch.cv, tokens, &w.cv, 4, dim, dim, 0.0);
        for r in 0..4 {
            let ck_row = &mut scratch.ck[r * dim..(r + 1) * dim];
            let cv_row = &mut scratch.cv[r * dim..(r + 1) * dim];
            for d in 0..dim {
                ck_row[d] += w.cbk[d];
                cv_row[d] += w.cbv[d];
            }
        }
        let attended = &mut scratch.ctx_attended[..dim];
        attention_flat(
            attended,
            &scratch.cq[..dim],
            &scratch.ck[..4 * dim],
            &scratch.cv[..4 * dim],
            Some(mask),
            dim,
            w.heads,
            &mut scratch.ctx_scores[..4],
        );
        matvec_bias(
            &mut scratch.ctx_projected,
            attended,
            &w.co_w,
            &w.co_b,
            dim,
            dim,
        );

        // ---- query residual + norm ----
        let query = &mut scratch.query[..dim];
        for d in 0..dim {
            query[d] = scratch.ctx_projected[d] + base_query[d];
        }
        layer_norm(query, &w.qn_w, &w.qn_b);

        // ---- audio attention over 41 frames ----
        matvec_bias(&mut scratch.aq, query, &w.aq, &w.abq, dim, dim);
        gemm_row_major(
            &mut scratch.ak,
            &scratch.audio,
            &w.ak,
            frames,
            dim,
            dim,
            0.0,
        );
        gemm_row_major(
            &mut scratch.av,
            &scratch.audio,
            &w.av,
            frames,
            dim,
            dim,
            0.0,
        );
        for t in 0..frames {
            let ak_row = &mut scratch.ak[t * dim..(t + 1) * dim];
            let av_row = &mut scratch.av[t * dim..(t + 1) * dim];
            for d in 0..dim {
                ak_row[d] += w.abk[d];
                av_row[d] += w.abv[d];
            }
        }
        let attended = &mut scratch.audio_attended[..dim];
        attention_flat(
            attended,
            &scratch.aq[..dim],
            &scratch.ak[..frames * dim],
            &scratch.av[..frames * dim],
            None,
            dim,
            w.heads,
            &mut scratch.audio_scores[..frames],
        );
        matvec_bias(
            &mut scratch.audio_projected,
            attended,
            &w.ao_w,
            &w.ao_b,
            dim,
            dim,
        );
        for d in 0..dim {
            query[d] += scratch.audio_projected[d];
        }
        layer_norm(query, &w.qn_w, &w.qn_b);

        // ---- audio norm (in-place) + activity/boundary logits + posterior mean ----
        let audio_norm = &mut scratch.audio[..frames * dim];
        for t in 0..frames {
            layer_norm(&mut audio_norm[t * dim..(t + 1) * dim], &w.an_w, &w.an_b);
        }
        let root = (dim as f32).sqrt();
        let mut activity = [0.0f32; FRAME_COUNT];
        for (t, value) in activity.iter_mut().enumerate() {
            let row = &audio_norm[t * dim..(t + 1) * dim];
            *value = dot(row, query) / root + dot(&w.act_w, row) + w.act_b;
        }
        let mut logits = [0.0f32; OFFSETS_MS.len()];
        for j in 0..OFFSETS_MS.len() {
            let row = &audio_norm[(14 + j) * dim..(15 + j) * dim];
            let local = dot(row, query) / root + dot(&w.bnd_w, row) + w.bnd_b;
            let derivative = activity[14 + j] - activity[14 + j - 1];
            let direction = if is_end == 0 { derivative } else { -derivative };
            logits[j] = local + w.derivative_scale * direction + w.prior[j];
        }
        softmax_mean(&logits, DECODER_TEMPERATURE)
    }
}

/// Reusable per-thread scratch for one endpoint's inference.  All buffers
/// are heap-allocated once per rayon worker thread and overwritten in place.
struct EndpointScratch {
    im1: Vec<f32>,
    im2: Vec<f32>,
    hidden_a: Vec<f32>,
    audio: Vec<f32>,
    tokens: Vec<f32>,
    base_query: Vec<f32>,
    cq: Vec<f32>,
    ck: Vec<f32>,
    cv: Vec<f32>,
    ctx_attended: Vec<f32>,
    ctx_projected: Vec<f32>,
    ctx_scores: Vec<f32>,
    query: Vec<f32>,
    aq: Vec<f32>,
    ak: Vec<f32>,
    av: Vec<f32>,
    audio_attended: Vec<f32>,
    audio_projected: Vec<f32>,
    audio_scores: Vec<f32>,
}

impl EndpointScratch {
    fn new(dim: usize) -> Self {
        let f = |n: usize| vec![0.0f32; n];
        Self {
            im1: f(FRAME_COUNT * 5 * FEATURE_DIM),
            im2: f(FRAME_COUNT * 5 * dim),
            hidden_a: f(FRAME_COUNT * dim),
            audio: f(FRAME_COUNT * dim),
            tokens: f(4 * dim),
            base_query: f(dim),
            cq: f(dim),
            ck: f(4 * dim),
            cv: f(4 * dim),
            ctx_attended: f(dim),
            ctx_projected: f(dim),
            ctx_scores: f(4),
            query: f(dim),
            aq: f(dim),
            ak: f(FRAME_COUNT * dim),
            av: f(FRAME_COUNT * dim),
            audio_attended: f(dim),
            audio_projected: f(dim),
            audio_scores: f(FRAME_COUNT),
        }
    }
}

/// `out[r][o] = Σ_k x[r][k] * w[o][k]`, row-major throughout.  Uses
/// `gemm::Parallelism::None` because callers already run under rayon.
fn gemm_row_major(out: &mut [f32], x: &[f32], w: &[f32], m: usize, n: usize, k: usize, beta: f32) {
    gemm_row_major_impl(out, x, w, m, n, k, beta, gemm::Parallelism::None);
}

/// Same as [`gemm_row_major`], parallelised across rayon for large outer calls.
fn gemm_row_major_par(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    m: usize,
    n: usize,
    k: usize,
    beta: f32,
) {
    gemm_row_major_impl(out, x, w, m, n, k, beta, gemm::Parallelism::Rayon(0));
}

fn gemm_row_major_impl(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    m: usize,
    n: usize,
    k: usize,
    beta: f32,
    parallelism: gemm::Parallelism,
) {
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            beta != 0.0,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
            k as isize,
            1,
            beta,
            1.0,
            false,
            false,
            false,
            parallelism,
        );
    }
}

/// Single-row linear layer `out[o] = dot(w[o], x) + bias[o]`, avoiding `gemm`
/// dispatch overhead for `m = 1` projections.
#[inline]
fn matvec_bias(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], n: usize, k: usize) {
    for o in 0..n {
        out[o] = dot(&w[o * k..(o + 1) * k], x) + bias[o];
    }
}

/// Flat-slice multi-head attention: one query, `n` keys/values packed as
/// `[n][dim]`, writing `[dim]`.  `scores` is caller-provided scratch `[n]`
/// (normalised in place, zero heap allocations).
fn attention_flat(
    out: &mut [f32],
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    mask: Option<&ContextMask>,
    dim: usize,
    heads: usize,
    scores: &mut [f32],
) {
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n = scores.len();
    for h in 0..heads {
        let q = &query[h * head_dim..(h + 1) * head_dim];
        for r in 0..n {
            let k = &keys[r * dim + h * head_dim..r * dim + (h + 1) * head_dim];
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += q[d] * k[d];
            }
            scores[r] = acc * scale;
            if let Some(mask) = mask {
                if mask[r] == 0.0 {
                    scores[r] = f32::NEG_INFINITY;
                }
            }
        }
        softmax_in_place(scores);
        let dst = &mut out[h * head_dim..(h + 1) * head_dim];
        dst.fill(0.0);
        for r in 0..n {
            let p = scores[r];
            let v = &values[r * dim + h * head_dim..r * dim + (h + 1) * head_dim];
            for d in 0..head_dim {
                dst[d] += p * v[d];
            }
        }
    }
}

/// Softmax at `temperature`, then the posterior mean over `OFFSETS_MS`,
/// computed without heap allocation.
fn softmax_mean(scores: &[f32], temperature: f32) -> f32 {
    let mut scaled = [0.0f32; OFFSETS_MS.len()];
    for (s, v) in scaled.iter_mut().zip(scores) {
        *s = *v / temperature;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in &scaled {
        max = max.max(v);
    }
    let mut total = 0.0f32;
    for v in scaled.iter_mut() {
        *v = (*v - max).exp();
        total += *v;
    }
    let mut mean = 0.0f32;
    for (p, o) in scaled.iter().zip(OFFSETS_MS) {
        mean += (*p / total) * o;
    }
    mean
}

// ---------------------------------------------------------------------------
// contract
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HeadFile {
    schema: String,
    architecture: String,
    offsets_ms: Vec<f64>,
    decoder_temperature: f64,
    decoder_amplitude: f64,
    inference_prior_scale: f64,
    context_roles: Vec<String>,
    feature_contract: FeatureContract,
    model_config: ModelConfigFile,
}

#[derive(Debug, Deserialize)]
struct FeatureContract {
    center_source: String,
    context_ms: u32,
    feature_dim: usize,
    frame_count: usize,
    frame_ms: usize,
    hop_ms: usize,
    n_fft: usize,
    n_mels: usize,
    sample_rate: usize,
    spectral_distance_bands: usize,
}

#[derive(Debug, Deserialize)]
struct ModelConfigFile {
    acoustic_dim: usize,
    activity_weight: f64,
    heads: usize,
    model_dim: usize,
    prior_sigma_ms: f64,
    qwen_dim: usize,
}

#[derive(Debug, Deserialize)]
struct NormalizerFile {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl HeadFile {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA,
            "local head schema {:?} != {SCHEMA:?}",
            self.schema
        );
        ensure!(
            self.architecture == ARCHITECTURE,
            "local head architecture {:?} != {ARCHITECTURE:?}",
            self.architecture
        );
        ensure!(
            self.context_roles.len() == CONTEXT_ROLES.len()
                && self
                    .context_roles
                    .iter()
                    .zip(CONTEXT_ROLES)
                    .all(|(a, b)| a == b),
            "local head context_roles {:?} != {CONTEXT_ROLES:?}",
            self.context_roles
        );
        ensure!(
            self.offsets_ms.len() == OFFSETS_MS.len()
                && self
                    .offsets_ms
                    .iter()
                    .zip(OFFSETS_MS)
                    .all(|(a, b)| (*a - b as f64).abs() <= 1e-6),
            "local head offsets_ms must be ±60 ms in 10 ms steps"
        );
        close(
            self.decoder_temperature,
            DECODER_TEMPERATURE as f64,
            "decoder_temperature",
        )?;
        close(self.decoder_amplitude, 1.0, "decoder_amplitude")?;
        close(
            self.inference_prior_scale,
            INFERENCE_PRIOR_SCALE,
            "inference_prior_scale",
        )?;

        let f = &self.feature_contract;
        ensure!(
            f.center_source == "fixed",
            "feature center_source must be \"fixed\""
        );
        ensure!(
            f.context_ms as usize == CONTEXT_MS,
            "feature context_ms must be {CONTEXT_MS}"
        );
        ensure!(
            f.feature_dim == FEATURE_DIM,
            "feature dim must be {FEATURE_DIM}"
        );
        ensure!(
            f.frame_count == FRAME_COUNT,
            "feature frame_count must be {FRAME_COUNT}"
        );
        ensure!(
            f.frame_ms == FRAME_MS,
            "feature frame_ms must be {FRAME_MS}"
        );
        ensure!(f.hop_ms == HOP_MS, "feature hop_ms must be {HOP_MS}");
        ensure!(f.n_fft == N_FFT, "feature n_fft must be {N_FFT}");
        ensure!(f.n_mels == N_MELS, "feature n_mels must be {N_MELS}");
        ensure!(
            f.sample_rate == SAMPLE_RATE,
            "feature sample_rate must be {SAMPLE_RATE}"
        );
        ensure!(
            f.spectral_distance_bands == SPECTRAL_BANDS,
            "feature spectral_distance_bands must be {SPECTRAL_BANDS}"
        );

        let m = &self.model_config;
        ensure!(
            m.acoustic_dim == FEATURE_DIM,
            "model_config.acoustic_dim must be {FEATURE_DIM}"
        );
        ensure!(m.qwen_dim == 1024, "model_config.qwen_dim must be 1024");
        ensure!(m.model_dim == 128, "model_config.model_dim must be 128");
        ensure!(m.heads == 4, "model_config.heads must be 4");
        close(
            m.prior_sigma_ms,
            PRIOR_SIGMA_MS,
            "model_config.prior_sigma_ms",
        )?;
        close(m.activity_weight, 0.25, "model_config.activity_weight")?;
        Ok(())
    }
}

fn close(a: f64, b: f64, what: &str) -> Result<()> {
    ensure!(
        (a - b).abs() <= 1e-6,
        "{what} {a} != the accepted head's {b}"
    );
    Ok(())
}

fn canonical_prior() -> Vec<f32> {
    OFFSETS_MS
        .iter()
        .map(|o| (-0.5 * (*o as f64 / PRIOR_SIGMA_MS).powi(2) * INFERENCE_PRIOR_SCALE) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

struct Weights {
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    ctx_w: Vec<f32>,
    ctx_b: Vec<f32>,
    role: Vec<f32>,
    endpoint: Vec<f32>,
    cq: Vec<f32>,
    ck: Vec<f32>,
    cv: Vec<f32>,
    cbq: Vec<f32>,
    cbk: Vec<f32>,
    cbv: Vec<f32>,
    co_w: Vec<f32>,
    co_b: Vec<f32>,
    aq: Vec<f32>,
    ak: Vec<f32>,
    av: Vec<f32>,
    abq: Vec<f32>,
    abk: Vec<f32>,
    abv: Vec<f32>,
    ao_w: Vec<f32>,
    ao_b: Vec<f32>,
    qn_w: Vec<f32>,
    qn_b: Vec<f32>,
    an_w: Vec<f32>,
    an_b: Vec<f32>,
    act_w: Vec<f32>,
    act_b: f32,
    bnd_w: Vec<f32>,
    bnd_b: f32,
    derivative_scale: f32,
    prior: Vec<f32>,
    model_dim: usize,
    heads: usize,
    context_in: usize,
}

impl Weights {
    fn numel(&self) -> usize {
        self.conv1_w.len()
            + self.conv1_b.len()
            + self.conv2_w.len()
            + self.conv2_b.len()
            + self.ctx_w.len()
            + self.ctx_b.len()
            + self.role.len()
            + self.endpoint.len()
            + self.cq.len()
            + self.ck.len()
            + self.cv.len()
            + self.cbq.len()
            + self.cbk.len()
            + self.cbv.len()
            + self.co_w.len()
            + self.co_b.len()
            + self.aq.len()
            + self.ak.len()
            + self.av.len()
            + self.abq.len()
            + self.abk.len()
            + self.abv.len()
            + self.ao_w.len()
            + self.ao_b.len()
            + self.qn_w.len()
            + self.qn_b.len()
            + self.an_w.len()
            + self.an_b.len()
            + self.act_w.len()
            + self.bnd_w.len()
            + 2
            + 1
            + self.prior.len()
    }

    fn load(path: &Path, cfg: &ModelConfigFile) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse {}", path.display()))?;
        let dim = cfg.model_dim;
        let qwen = cfg.qwen_dim;

        // `as_chunks` is what clippy suggests, and it needs a newer MSRV.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        let get = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let tensor = st
                .tensor(name)
                .with_context(|| format!("local head tensor {name}"))?;
            ensure!(
                tensor.dtype() == Dtype::F32,
                "local head tensor {name} is {:?}, expected F32",
                tensor.dtype()
            );
            ensure!(
                tensor.shape() == shape,
                "local head tensor {name} is {:?}, expected {shape:?}",
                tensor.shape()
            );
            Ok(tensor
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let scalar = |name: &str| -> Result<f32> {
            let tensor = st
                .tensor(name)
                .with_context(|| format!("local head tensor {name}"))?;
            ensure!(
                tensor.dtype() == Dtype::F32,
                "local head tensor {name} must be F32"
            );
            ensure!(
                tensor.data().len() == 4,
                "local head tensor {name} must be scalar"
            );
            Ok(f32::from_le_bytes([
                tensor.data()[0],
                tensor.data()[1],
                tensor.data()[2],
                tensor.data()[3],
            ]))
        };
        // `nn.MultiheadAttention` packs q, k, v into one [3*dim, dim] matrix.
        type InProj = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
        let split = |name: &str| -> Result<InProj> {
            let w = get(&format!("{name}.in_proj_weight"), &[3 * dim, dim])?;
            let b = get(&format!("{name}.in_proj_bias"), &[3 * dim])?;
            let block = dim * dim;
            Ok((
                w[..block].to_vec(),
                w[block..2 * block].to_vec(),
                w[2 * block..].to_vec(),
                b[..dim].to_vec(),
                b[dim..2 * dim].to_vec(),
                b[2 * dim..].to_vec(),
            ))
        };

        let (cq, ck, cv, cbq, cbk, cbv) = split("context_attention")?;
        let (aq, ak, av, abq, abk, abv) = split("audio_attention")?;

        // [c][ic][k] -> [c][k][ic]: each output channel's 5 taps become a
        // contiguous [5, in] block so interior im2col rows are a memcpy.
        let prepack_conv = |raw: &[f32], out: usize, inc: usize| -> Vec<f32> {
            let mut packed = vec![0.0f32; out * 5 * inc];
            for c in 0..out {
                for k in 0..5 {
                    for ic in 0..inc {
                        packed[c * 5 * inc + k * inc + ic] = raw[(c * inc + ic) * 5 + k];
                    }
                }
            }
            packed
        };
        let conv1_raw = get("audio.0.weight", &[dim, FEATURE_DIM, 5])?;
        let conv2_raw = get("audio.2.weight", &[dim, dim, 5])?;
        let conv1_w = prepack_conv(&conv1_raw, dim, FEATURE_DIM);
        let conv2_w = prepack_conv(&conv2_raw, dim, dim);

        Ok(Self {
            conv1_w,
            conv1_b: get("audio.0.bias", &[dim])?,
            conv2_w,
            conv2_b: get("audio.2.bias", &[dim])?,
            ctx_w: get("context_projection.weight", &[dim, qwen])?,
            ctx_b: get("context_projection.bias", &[dim])?,
            role: get("role_embedding.weight", &[4, dim])?,
            endpoint: get("endpoint_embedding.weight", &[2, dim])?,
            cq,
            ck,
            cv,
            cbq,
            cbk,
            cbv,
            co_w: get("context_attention.out_proj.weight", &[dim, dim])?,
            co_b: get("context_attention.out_proj.bias", &[dim])?,
            aq,
            ak,
            av,
            abq,
            abk,
            abv,
            ao_w: get("audio_attention.out_proj.weight", &[dim, dim])?,
            ao_b: get("audio_attention.out_proj.bias", &[dim])?,
            qn_w: get("query_norm.weight", &[dim])?,
            qn_b: get("query_norm.bias", &[dim])?,
            an_w: get("audio_norm.weight", &[dim])?,
            an_b: get("audio_norm.bias", &[dim])?,
            act_w: get("activity_bias.weight", &[1, dim])?,
            act_b: get("activity_bias.bias", &[1])?[0],
            bnd_w: get("boundary_bias.weight", &[1, dim])?,
            bnd_b: get("boundary_bias.bias", &[1])?[0],
            derivative_scale: scalar("derivative_scale")?,
            prior: get("qwen_prior", &[13])?,
            model_dim: dim,
            heads: cfg.heads,
            context_in: qwen,
        })
    }
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

/// `_mel_filterbank`: HTK-style mel points, floored bin edges, per-row unit sum.
fn mel_filterbank() -> Vec<f32> {
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let low = hz_to_mel(0.0);
    let high = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let bins = N_FFT / 2 + 1;
    let points: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(low + (high - low) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let edges: Vec<usize> = points
        .iter()
        .map(|p| (((N_FFT + 1) as f64 * p / SAMPLE_RATE as f64).floor() as i64).max(0) as usize)
        .collect();

    let mut bank = vec![0.0f32; N_MELS * bins];
    for m in 0..N_MELS {
        let left = edges[m];
        let center = edges[m + 1].max(left + 1);
        let right = edges[m + 2].max(center + 1).min(bins - 1);
        if center >= bins {
            continue;
        }
        if center > left {
            let n = center - left;
            for i in 0..n {
                bank[m * bins + left + i] = (i as f32) / n as f32;
            }
        }
        if right > center {
            let n = right - center;
            for i in 0..n {
                bank[m * bins + center + i] = 1.0 - (i as f32) / n as f32;
            }
        }
        let mut total = 0.0f32;
        for b in 0..bins {
            total += bank[m * bins + b];
        }
        if total > 0.0 {
            for b in 0..bins {
                bank[m * bins + b] /= total;
            }
        }
    }
    bank
}

fn mel_active_ranges(bank: &[f32]) -> Vec<(usize, usize)> {
    let bins = N_FFT / 2 + 1;
    (0..N_MELS)
        .map(|m| {
            let row = &bank[m * bins..(m + 1) * bins];
            let start = row.iter().position(|v| *v != 0.0).unwrap_or(bins);
            let end = row.iter().rposition(|v| *v != 0.0).map_or(start, |i| i + 1);
            (start, end)
        })
        .collect()
}

/// `_band_edges`: four contiguous bands over the 257 rfft bins.
fn band_edges(bins: usize) -> [usize; SPECTRAL_BANDS + 1] {
    let mut edges = [0usize; SPECTRAL_BANDS + 1];
    for (i, e) in edges.iter_mut().enumerate() {
        let v = 1.0 + (bins as f64 - 1.0) * i as f64 / SPECTRAL_BANDS as f64;
        *e = v as usize;
    }
    edges
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32]) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..n {
        x[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

fn softmax_in_place(scores: &mut [f32]) {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for v in scores.iter_mut() {
        *v = (*v - max).exp();
        total += *v;
    }
    for v in scores.iter_mut() {
        *v /= total;
    }
}

// ---------------------------------------------------------------------------
// coarse stream and interval repair
// ---------------------------------------------------------------------------

/// `_timestamp_valid_class_count`: classes whose coarse time fits the audio.
fn valid_class_count(duration_ms: f64, class_count: usize) -> usize {
    let by_audio = (duration_ms / SEGMENT_MS).floor() as i64 + 1;
    by_audio.clamp(1, class_count as i64) as usize
}

fn argmax_ms(row: &[f32]) -> i64 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    (best as f64 * SEGMENT_MS) as i64
}

/// `_remove_neighbor_overlaps`: trust the next word's start, trim this end.
fn remove_neighbor_overlaps(intervals: &mut [[f64; 2]]) {
    for index in 0..intervals.len().saturating_sub(1) {
        if intervals[index][1] <= intervals[index + 1][0] {
            continue;
        }
        let mut cut = intervals[index + 1][0];
        let minimum_cut = intervals[index][0] + MIN_WORD_DURATION_MS;
        if cut < minimum_cut {
            cut = minimum_cut;
            intervals[index + 1][0] = cut;
            intervals[index + 1][1] = intervals[index + 1][1].max(cut + MIN_WORD_DURATION_MS);
        }
        intervals[index][1] = cut;
    }
}

/// `_bound_intervals_to_duration`: clamp, keeping 10 ms per word where the
/// audio length allows it.
fn bound_to_duration(intervals: &mut [[f64; 2]], duration_ms: f64) {
    if intervals.is_empty() {
        return;
    }
    let count = intervals.len();
    let minimum = MIN_WORD_DURATION_MS.min(duration_ms / count as f64);
    if intervals.iter().all(|i| {
        i[0].is_finite() && i[1].is_finite() && i[0] >= 0.0 && i[0] < i[1] && i[1] <= duration_ms
    }) {
        return;
    }
    let mut cursor = 0.0f64;
    for (index, interval) in intervals.iter_mut().enumerate() {
        let mut start = interval[0];
        let mut end = interval[1];
        if !start.is_finite() || !end.is_finite() {
            start = 0.0;
            end = duration_ms;
        }
        let remaining = (count - index - 1) as f64;
        let latest_start = duration_ms - minimum * (remaining + 1.0);
        start = start.max(cursor).min(latest_start);
        let earliest_end = start + minimum;
        let latest_end = duration_ms - minimum * remaining;
        end = end.max(earliest_end).min(latest_end);
        *interval = [start, end];
        cursor = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_matches_numpy_hanning() {
        // np.hanning(400).astype(np.float32)[0, 100, 199, 200, 399]
        // -> 0.0, 0.5019683837890625, 0.9999845027923584, 0.9999845027923584, 0.0
        // (an even-length Hann has its peak split between the two middle samples)
        let w = hann_window(400);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[399], 0.0);
        assert!((w[100] - 0.5019684).abs() < 1e-6, "{}", w[100]);
        assert!((w[199] - 0.9999845).abs() < 1e-6, "{}", w[199]);
        assert!((w[200] - 0.9999845).abs() < 1e-6, "{}", w[200]);
        assert!(w.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!((w[10] - w[389]).abs() < 1e-6, "the window is symmetric");
    }

    #[test]
    fn mel_bank_rows_are_unit_sum_with_a_triangular_shape() {
        let bank = mel_filterbank();
        let bins = N_FFT / 2 + 1;
        for m in 0..N_MELS {
            let total: f32 = bank[m * bins..(m + 1) * bins].iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "row {m} sums to {total}");
            assert!(bank[m * bins..(m + 1) * bins].iter().all(|v| *v >= 0.0));
        }
    }

    #[test]
    fn bands_cover_the_spectrum_contiguously() {
        assert_eq!(band_edges(257), [1, 65, 129, 193, 257]);
    }

    #[test]
    fn duration_masking_truncates_to_the_audio_length() {
        // 2.3811875 s of audio -> floor(2381.1875 / 80) + 1 = 30 classes
        assert_eq!(valid_class_count(2381.1875, 5000), 30);
        assert_eq!(valid_class_count(0.0, 5000), 1);
        assert_eq!(valid_class_count(1e9, 5000), 5000);
    }

    #[test]
    fn overlap_repair_trusts_the_next_start() {
        let mut v = vec![[100.0, 300.0], [250.0, 400.0]];
        remove_neighbor_overlaps(&mut v);
        assert_eq!(v, vec![[100.0, 250.0], [250.0, 400.0]]);

        // too short to cut: move the next start instead
        let mut v = vec![[100.0, 300.0], [104.0, 200.0]];
        remove_neighbor_overlaps(&mut v);
        assert_eq!(v, vec![[100.0, 110.0], [110.0, 200.0]]);
    }

    #[test]
    fn duration_bound_keeps_minimum_lengths_when_it_can() {
        // already inside the duration: untouched
        let mut v = vec![[0.0, 5.0], [5.0, 8.0]];
        bound_to_duration(&mut v, 10.0);
        assert_eq!(v, vec![[0.0, 5.0], [5.0, 8.0]]);

        // second end past the audio: both ends are pulled inside, each word
        // keeping `min(10 ms, duration / n) = 5 ms`
        let mut v = vec![[0.0, 5.0], [9.0, 12.0]];
        bound_to_duration(&mut v, 10.0);
        assert!((v[1][1] - 10.0).abs() < 1e-9, "{:?}", v);
        assert!(v[0][1] <= v[1][0] + 1e-9);
        assert!(v[0][1] - v[0][0] >= 5.0 - 1e-9);
    }

    // ---------------------------------------------------------------------
    // local fast-path equivalence tests
    // ---------------------------------------------------------------------

    /// A head with a real normalizer/bank/FFT but zero weights — sufficient
    /// for feature extraction tests, which don't touch the network weights.
    fn synth_head() -> LocalHead {
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let fft_scratch_len = fft.get_inplace_scratch_len();
        let bank = mel_filterbank();
        let mel_ranges = mel_active_ranges(&bank);
        let dim = 128;
        let zeros = |n: usize| vec![0.0f32; n];
        LocalHead {
            weights: Weights {
                conv1_w: zeros(dim * FEATURE_DIM * 5),
                conv1_b: zeros(dim),
                conv2_w: zeros(dim * dim * 5),
                conv2_b: zeros(dim),
                ctx_w: zeros(dim * 1024),
                ctx_b: zeros(dim),
                role: zeros(4 * dim),
                endpoint: zeros(2 * dim),
                cq: zeros(dim * dim),
                ck: zeros(dim * dim),
                cv: zeros(dim * dim),
                cbq: zeros(dim),
                cbk: zeros(dim),
                cbv: zeros(dim),
                co_w: zeros(dim * dim),
                co_b: zeros(dim),
                aq: zeros(dim * dim),
                ak: zeros(dim * dim),
                av: zeros(dim * dim),
                abq: zeros(dim),
                abk: zeros(dim),
                abv: zeros(dim),
                ao_w: zeros(dim * dim),
                ao_b: zeros(dim),
                qn_w: vec![1.0; dim],
                qn_b: zeros(dim),
                an_w: vec![1.0; dim],
                an_b: zeros(dim),
                act_w: zeros(dim),
                act_b: 0.0,
                bnd_w: zeros(dim),
                bnd_b: 0.0,
                derivative_scale: 0.5,
                prior: canonical_prior(),
                model_dim: dim,
                heads: 4,
                context_in: 1024,
            },
            mean: vec![0.0; FEATURE_DIM],
            std: vec![1.0; FEATURE_DIM],
            fft,
            hann: hann_window(FRAME_MS * SAMPLE_RATE / 1000),
            bank,
            mel_ranges,
            fft_scratch_len,
        }
    }

    #[test]
    fn global_features_match_features_one() {
        let head = synth_head();
        // ~0.5 s of deterministic pseudo-audio, enough for boundary frames.
        let samples: Vec<f32> = (0..8_000)
            .map(|i| ((i * 7919 % 4093) as f32 / 4093.0 - 0.5) * 0.2)
            .collect();
        let centers: Vec<i64> = vec![
            0,  // leading boundary
            10, // overlapping with the next
            20, 80,  // coarse grid
            320, // interior
            480, // near the end of audio (500 ms)
        ];
        let global = head.features(&samples, &centers);
        let mut worst = 0.0f32;
        for (k, &center) in centers.iter().enumerate() {
            let mut one = vec![0.0f32; FRAME_COUNT * FEATURE_DIM];
            head.features_one(&samples, center, &mut one);
            let got = &global[k * FRAME_COUNT * FEATURE_DIM..(k + 1) * FRAME_COUNT * FEATURE_DIM];
            for (a, b) in one.iter().zip(got) {
                worst = worst.max((a - b).abs());
            }
        }
        eprintln!("global vs one feature max |diff| = {worst:.3e}");
        assert!(worst < 1e-5, "global features diverge: {worst}");
        assert!(head.features(&samples, &[]).is_empty());
    }

    #[test]
    fn off_grid_centers_fall_back_to_features_one() {
        let head = synth_head();
        let samples: Vec<f32> = (0..8_000)
            .map(|i| ((i * 104729 % 4093) as f32 / 4093.0 - 0.5) * 0.2)
            .collect();
        // 15 ms is off the 10 ms grid; the fallback path must be used.
        let centers = [15i64, 105, 255];
        let got = head.features(&samples, &centers);
        for (k, &center) in centers.iter().enumerate() {
            let mut one = vec![0.0f32; FRAME_COUNT * FEATURE_DIM];
            head.features_one(&samples, center, &mut one);
            assert_eq!(
                got[k * FRAME_COUNT * FEATURE_DIM..(k + 1) * FRAME_COUNT * FEATURE_DIM],
                one,
                "off-grid center {center}"
            );
        }
    }

    #[test]
    fn batched_refine_matches_single_endpoint_offset() {
        let Some(g) = golden() else { return };
        let samples = f32_file(
            &g.dir
                .join(g.meta["audio_file"].as_str().expect("audio_file")),
        );
        let logits = f32_file(
            &g.dir
                .join(g.meta["logits_file"].as_str().expect("logits_file")),
        );
        let hidden = f32_file(
            &g.dir
                .join(g.meta["hidden_file"].as_str().expect("hidden_file")),
        );
        let n_labels = g.meta["num_labels"].as_u64().expect("num_labels") as usize;
        let hidden_size = g.meta["hidden_shape"][0].as_u64().expect("hidden") as usize;
        let slots = hidden.len() / hidden_size;
        let duration_ms = samples.len() as f64 * 1000.0 / SAMPLE_RATE as f64;
        let valid = valid_class_count(duration_ms, n_labels);
        let raw: Vec<i64> = (0..slots)
            .map(|s| argmax_ms(&logits[s * n_labels..s * n_labels + valid]))
            .collect();
        let fixed = fix_timestamps(&raw);
        let (context, masks) = g.head.context_rows(&hidden, hidden_size, slots / 2);
        let features = g.head.features(&samples, &fixed);

        // Compare `refine_offsets` (pre-projected unique slot/word-mean GEMM)
        // against the per-endpoint `context_rows` + `endpoint_offset` path.
        let batched = g.head.refine_offsets(&features, &hidden, hidden_size, slots / 2);
        let mut worst = 0.0f32;
        for e in 0..slots {
            let feat =
                &features[e * FRAME_COUNT * FEATURE_DIM..(e + 1) * FRAME_COUNT * FEATURE_DIM];
            let ctx = &context[e * 4 * hidden_size..(e + 1) * 4 * hidden_size];
            let single = g.head.endpoint_offset(feat, ctx, &masks[e], e % 2);
            worst = worst.max((single - batched[e]).abs());
        }
        eprintln!("batched vs single endpoint max |diff| = {worst:.3e} ms");
        assert!(worst < 1e-4, "batched offsets diverge: {worst} ms");
    }

    // ---------------------------------------------------------------------
    // parity against the python reference
    //
    // `tools/gold/local_head/15s_en.*` is the frozen python run (gitignored,
    // like the other golds): the exact samples it read, the timestamp logits
    // and final-norm rows it saw, and the offsets/items it produced.  The
    // generator is local tooling; the JSON contract is the field names used
    // below.  The audio sidecar replaces the wav on purpose — the wav decode
    // and the resampler are not part of this contract.
    // ---------------------------------------------------------------------

    struct Golden {
        dir: PathBuf,
        meta: serde_json::Value,
        head: LocalHead,
    }

    fn golden() -> Option<Golden> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/gold/local_head");
        let file = dir.join("15s_en.json");
        if !file.is_file() {
            eprintln!("skip: {file:?} is absent (see the note above on the frozen gold)");
            return None;
        }
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).ok()?).ok()?;
        let model = PathBuf::from(meta["model"].as_str()?);
        let head = match LocalHead::load(&model) {
            Ok(Some(h)) => h,
            Ok(None) => {
                eprintln!("skip: {model:?} has no local_head/");
                return None;
            }
            Err(e) => {
                eprintln!("skip: {model:?} does not hold a valid local_head/: {e}");
                return None;
            }
        };
        Some(Golden { dir, meta, head })
    }

    fn f32_file(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .expect("sidecar")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn nums(meta: &serde_json::Value, key: &str) -> Vec<f64> {
        meta[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} must be an array"))
            .iter()
            .map(|v| v.as_f64().expect("number"))
            .collect()
    }

    #[test]
    fn features_match_python_golden() {
        let Some(g) = golden() else { return };
        let samples = f32_file(
            &g.dir
                .join(g.meta["audio_file"].as_str().expect("audio_file")),
        );
        assert_eq!(
            samples.len(),
            g.meta["audio_samples"].as_u64().expect("len") as usize
        );
        let fixed: Vec<i64> = nums(&g.meta, "fixed_ms")
            .iter()
            .map(|v| *v as i64)
            .collect();
        let all = g.head.features(&samples, &fixed);

        let sample: Vec<usize> = g.meta["sample_endpoints"]
            .as_array()
            .expect("sample_endpoints")
            .iter()
            .map(|v| v.as_u64().expect("index") as usize)
            .collect();
        let want = g.meta["features"].as_array().expect("features");
        assert_eq!(want.len(), sample.len());

        let mut worst = 0.0f32;
        for (k, &endpoint) in sample.iter().enumerate() {
            let got = &all
                [endpoint * FRAME_COUNT * FEATURE_DIM..(endpoint + 1) * FRAME_COUNT * FEATURE_DIM];
            let row = want[k].as_array().expect("row");
            assert_eq!(row.len(), got.len());
            for (i, v) in row.iter().enumerate() {
                worst = worst.max((v.as_f64().expect("value") as f32 - got[i]).abs());
            }
        }
        eprintln!("feature max |diff| = {worst:.3e}");
        assert!(worst < 2e-3, "features diverge from the reference: {worst}");
    }

    #[test]
    fn offsets_match_python_golden() {
        let Some(g) = golden() else { return };
        let sample: Vec<usize> = g.meta["sample_endpoints"]
            .as_array()
            .expect("sample_endpoints")
            .iter()
            .map(|v| v.as_u64().expect("index") as usize)
            .collect();
        let features = g.meta["features"].as_array().expect("features");
        let context = g.meta["context"].as_array().expect("context");
        let mask = g.meta["mask"].as_array().expect("mask");
        let endpoint = g.meta["endpoint"].as_array().expect("endpoint");
        let offsets = nums(&g.meta, "offsets_ms");

        let mut worst = 0.0f32;
        for (k, &index) in sample.iter().enumerate() {
            let feat: Vec<f32> = features[k]
                .as_array()
                .expect("features row")
                .iter()
                .map(|v| v.as_f64().expect("f") as f32)
                .collect();
            let ctx: Vec<f32> = context[k]
                .as_array()
                .expect("context row")
                .iter()
                .map(|v| v.as_f64().expect("f") as f32)
                .collect();
            let m: ContextMask = [0, 1, 2, 3].map(|i| mask[k][i].as_f64().expect("mask") as f32);
            let e = endpoint[k].as_u64().expect("endpoint") as usize;
            let got = g.head.endpoint_offset(&feat, &ctx, &m, e);
            let want = offsets[index] as f32;
            worst = worst.max((got - want).abs());
            eprintln!(
                "endpoint {index:3} ({e}): rust {got:9.4}  python {want:9.4}  d {:.4}",
                got - want
            );
        }
        eprintln!("offset max |diff| = {worst:.4} ms");
        assert!(
            worst < 0.05,
            "head offsets diverge from the reference: {worst} ms"
        );
    }

    #[test]
    fn refine_matches_python_golden() {
        let Some(g) = golden() else { return };
        let samples = f32_file(
            &g.dir
                .join(g.meta["audio_file"].as_str().expect("audio_file")),
        );
        let logits = f32_file(
            &g.dir
                .join(g.meta["logits_file"].as_str().expect("logits_file")),
        );
        let hidden = f32_file(
            &g.dir
                .join(g.meta["hidden_file"].as_str().expect("hidden_file")),
        );
        let n_labels = g.meta["num_labels"].as_u64().expect("num_labels") as usize;
        let hidden_size = g.meta["hidden_shape"][0].as_u64().expect("hidden") as usize;
        let words: Vec<String> = g.meta["words"]
            .as_array()
            .expect("words")
            .iter()
            .map(|v| v.as_str().expect("word").to_string())
            .collect();

        let items = g
            .head
            .refine(&samples, &words, &logits, n_labels, &hidden, hidden_size)
            .expect("refine");
        let want = g.meta["items"].as_array().expect("items");
        assert_eq!(items.len(), want.len());

        let mut worst = 0.0f64;
        for (i, item) in items.iter().enumerate() {
            assert_eq!(item.text, want[i][0].as_str().expect("text"));
            for (got, key) in [(item.start_time, 1), (item.end_time, 2)] {
                let want_v = want[i][key].as_f64().expect("time");
                let d = (got - want_v).abs() * 1000.0;
                if d > 0.5 {
                    eprintln!(
                        "word {i:2} {:12} {:>5}: rust {got:.4}  python {want_v:.4}  d {d:.3} ms",
                        item.text,
                        ["start", "end"][key - 1]
                    );
                }
                worst = worst.max(d);
            }
        }
        eprintln!("refine max |diff| = {worst:.4} ms");
        assert!(
            worst < 0.5,
            "refined intervals diverge from the reference: {worst} ms"
        );
    }
}
