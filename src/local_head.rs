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
        Ok(Some(Self {
            weights,
            mean: normalizer.mean,
            std: normalizer.std,
            fft: planner.plan_fft_forward(N_FFT),
            hann: hann_window(FRAME_MS * SAMPLE_RATE / 1000),
            bank: mel_filterbank(),
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

        // 2. context rows, four per endpoint
        let (context, masks) = self.context_rows(hidden, hidden_size, word_count);

        // 3. the 91-channel window centred on each fixed endpoint
        let features = self.features(samples, &fixed_ms);

        // 4. per-endpoint offsets — independent, so plain data parallelism
        let offsets: Vec<f32> = (0..slots)
            .into_par_iter()
            .map(|endpoint| {
                let feat = &features
                    [endpoint * FRAME_COUNT * FEATURE_DIM..(endpoint + 1) * FRAME_COUNT * FEATURE_DIM];
                let ctx =
                    &context[endpoint * 4 * hidden_size..(endpoint + 1) * 4 * hidden_size];
                self.endpoint_offset(feat, ctx, &masks[endpoint], endpoint % 2)
            })
            .collect();

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

    /// `_context_for_sample`: `[slot, current avg, prev avg, next avg]` per
    /// endpoint, with the missing neighbours at the edges as zeros but flagged
    /// in the mask.
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
            let previous = if w > 0 { mean(w * 2 - 2, w * 2) } else { vec![0.0; hidden_size] };
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
                    let dst = &mut context[base + role * hidden_size..base + (role + 1) * hidden_size];
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
        out.par_chunks_mut(FRAME_COUNT * FEATURE_DIM)
            .zip(centers_ms.par_iter())
            .for_each(|(dst, &center_ms)| self.features_one(samples, center_ms, dst));
        out
    }

    fn features_one(&self, samples: &[f32], center_ms: i64, dst: &mut [f32]) {
        let frame_len = FRAME_MS * SAMPLE_RATE / 1000; // 400
        let hop = HOP_MS * SAMPLE_RATE / 1000; // 160
        let bins = N_FFT / 2 + 1; // 257
        let center_sample = center_ms * (SAMPLE_RATE as i64) / 1000;

        // 41 hann-windowed frames, zero-padded outside the audio
        let mut frames = vec![0.0f32; FRAME_COUNT * frame_len];
        for f in 0..FRAME_COUNT {
            let start = center_sample
                + (f as i64 - (CONTEXT_MS / HOP_MS) as i64) * hop as i64
                - (frame_len / 2) as i64;
            let base = f * frame_len;
            for i in 0..frame_len {
                let index = start + i as i64;
                if index >= 0 && (index as usize) < samples.len() {
                    frames[base + i] = samples[index as usize] * self.hann[i];
                }
            }
        }

        // |X| per frame.  numpy's rfft widens to f64, so the transform is f64
        // here too and only the magnitude is narrowed back to f32.
        let mut magnitude = vec![0.0f32; FRAME_COUNT * bins];
        let mut buffer = vec![Complex::new(0.0f64, 0.0); N_FFT];
        for f in 0..FRAME_COUNT {
            buffer.iter_mut().for_each(|b| *b = Complex::new(0.0, 0.0));
            for i in 0..frame_len {
                buffer[i].re = frames[f * frame_len + i] as f64;
            }
            self.fft.process(&mut buffer);
            for b in 0..bins {
                magnitude[f * bins + b] = buffer[b].norm() as f32;
            }
        }

        let mut power = vec![0.0f32; FRAME_COUNT * bins];
        for (p, m) in power.iter_mut().zip(&magnitude) {
            *p = *m * *m;
        }

        let mut log_mel = vec![0.0f32; FRAME_COUNT * N_MELS];
        for f in 0..FRAME_COUNT {
            for m in 0..N_MELS {
                let bank = &self.bank[m * bins..(m + 1) * bins];
                let mut acc = 0.0f32;
                for b in 0..bins {
                    acc += bank[b] * power[f * bins + b];
                }
                log_mel[f * N_MELS + m] = acc.max(LOG_EPSILON).ln();
            }
        }

        let mut log_rms = [0.0f32; FRAME_COUNT];
        let mut delta_rms = [0.0f32; FRAME_COUNT];
        for f in 0..FRAME_COUNT {
            let frame = &frames[f * frame_len..(f + 1) * frame_len];
            let mut acc = 0.0f32;
            for v in frame {
                acc += v * v;
            }
            log_rms[f] = (acc / frame_len as f32 + LOG_EPSILON)
                .sqrt()
                .max(LOG_EPSILON)
                .ln();
            delta_rms[f] = if f == 0 { 0.0 } else { log_rms[f] - log_rms[f - 1] };
        }

        // Mean-removed log spectrum, which the flux and distance channels read.
        let mut normalised = vec![0.0f32; FRAME_COUNT * bins];
        for f in 0..FRAME_COUNT {
            let mut mean = 0.0f32;
            for b in 0..bins {
                let v = magnitude[f * bins + b].max(LOG_EPSILON).ln();
                normalised[f * bins + b] = v;
                mean += v;
            }
            mean /= bins as f32;
            for b in 0..bins {
                normalised[f * bins + b] -= mean;
            }
        }

        let mut flux = [0.0f32; FRAME_COUNT];
        for f in 1..FRAME_COUNT {
            let prev = &magnitude[(f - 1) * bins..f * bins];
            let cur = &magnitude[f * bins..(f + 1) * bins];
            let mut positive = 0.0f32;
            let mut total = 0.0f32;
            for b in 0..bins {
                positive += (cur[b] - prev[b]).max(0.0);
                total += prev[b];
            }
            flux[f] = positive / (total + LOG_EPSILON);
        }

        // Spectral distance to the previous/next frame inside four bands.
        let edges = band_edges(bins);
        let mut distance = vec![0.0f32; FRAME_COUNT * SPECTRAL_BANDS * 2];
        for band in 0..SPECTRAL_BANDS {
            let left = edges[band];
            let right = edges[band + 1].max(left + 1);
            for f in 0..FRAME_COUNT {
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
                distance[f * SPECTRAL_BANDS * 2 + band] = acc_l / width;
                distance[f * SPECTRAL_BANDS * 2 + SPECTRAL_BANDS + band] = acc_r / width;
            }
        }

        // [41][91]: 80 log-mel, log-RMS, delta, flux, 8 distances — then the
        // train-stat normalizer, channel by channel.
        for f in 0..FRAME_COUNT {
            let row = &mut dst[f * FEATURE_DIM..(f + 1) * FEATURE_DIM];
            row[..N_MELS].copy_from_slice(&log_mel[f * N_MELS..(f + 1) * N_MELS]);
            row[N_MELS] = log_rms[f];
            row[N_MELS + 1] = delta_rms[f];
            row[N_MELS + 2] = flux[f];
            row[N_MELS + 3..]
                .copy_from_slice(&distance[f * SPECTRAL_BANDS * 2..(f + 1) * SPECTRAL_BANDS * 2]);
            for ((value, mean), std) in row.iter_mut().zip(&self.mean).zip(&self.std) {
                *value = (*value - *mean) / *std;
            }
        }
    }

    /// One endpoint through `IdentityConditionedBoundaryHead`; the decoded
    /// offset in milliseconds.
    fn endpoint_offset(&self, feat: &[f32], ctx: &[f32], mask: &ContextMask, is_end: usize) -> f32 {
        let w = &self.weights;
        let dim = w.model_dim;

        // conv1: [91][41] -> [128][41] -> GELU; conv2 over the 128 channels
        let mut hidden_a = vec![0.0f32; dim * FRAME_COUNT];
        for c in 0..dim {
            for t in 0..FRAME_COUNT {
                let mut acc = w.conv1_b[c];
                for k in 0..5 {
                    let tt = t as isize + k as isize - 2;
                    if !(0..FRAME_COUNT as isize).contains(&tt) {
                        continue;
                    }
                    let tt = tt as usize;
                    for ic in 0..FEATURE_DIM {
                        acc += w.conv1_w[(c * FEATURE_DIM + ic) * 5 + k] * feat[tt * FEATURE_DIM + ic];
                    }
                }
                hidden_a[c * FRAME_COUNT + t] = gelu(acc);
            }
        }
        let mut hidden_b = vec![0.0f32; dim * FRAME_COUNT];
        for c in 0..dim {
            for t in 0..FRAME_COUNT {
                let mut acc = w.conv2_b[c];
                for k in 0..5 {
                    let tt = t as isize + k as isize - 2;
                    if !(0..FRAME_COUNT as isize).contains(&tt) {
                        continue;
                    }
                    let tt = tt as usize;
                    for ic in 0..dim {
                        acc += w.conv2_w[(c * dim + ic) * 5 + k] * hidden_a[ic * FRAME_COUNT + tt];
                    }
                }
                hidden_b[c * FRAME_COUNT + t] = gelu(acc);
            }
        }
        // [41][128], frame-major, which is the layout attention wants
        let mut audio = vec![0.0f32; FRAME_COUNT * dim];
        for t in 0..FRAME_COUNT {
            for d in 0..dim {
                audio[t * dim + d] = hidden_b[d * FRAME_COUNT + t];
            }
        }

        // context projection + role embedding
        let mut tokens = vec![0.0f32; 4 * dim];
        for r in 0..4 {
            for d in 0..dim {
                let mut acc = w.ctx_b[d] + w.role[r * dim + d];
                for j in 0..w.context_in {
                    acc += w.ctx_w[d * w.context_in + j] * ctx[r * w.context_in + j];
                }
                tokens[r * dim + d] = acc;
            }
        }
        let base_query: Vec<f32> = (0..dim).map(|d| tokens[d] + w.endpoint[is_end * dim + d]).collect();

        // context attention: one query, four masked keys
        let proj_row = |x: &[f32], weight: &[f32], bias: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; dim];
            for d in 0..dim {
                let mut acc = bias[d];
                for j in 0..dim {
                    acc += weight[d * dim + j] * x[j];
                }
                out[d] = acc;
            }
            out
        };
        let q = proj_row(&base_query, &w.cq, &w.cbq);
        let keys: Vec<Vec<f32>> = (0..4).map(|r| proj_row(&tokens[r * dim..(r + 1) * dim], &w.ck, &w.cbk)).collect();
        let values: Vec<Vec<f32>> = (0..4).map(|r| proj_row(&tokens[r * dim..(r + 1) * dim], &w.cv, &w.cbv)).collect();
        let attended = attention(&q, &keys, &values, Some(mask), dim, w.heads);
        let projected = proj_row(&attended, &w.co_w, &w.co_b);

        // residual + query_norm, then attention over the 41 acoustic frames
        let mut query: Vec<f32> = (0..dim).map(|d| projected[d] + base_query[d]).collect();
        layer_norm(&mut query, &w.qn_w, &w.qn_b);

        let q = proj_row(&query, &w.aq, &w.abq);
        let keys: Vec<Vec<f32>> = (0..FRAME_COUNT)
            .map(|t| proj_row(&audio[t * dim..(t + 1) * dim], &w.ak, &w.abk))
            .collect();
        let values: Vec<Vec<f32>> = (0..FRAME_COUNT)
            .map(|t| proj_row(&audio[t * dim..(t + 1) * dim], &w.av, &w.abv))
            .collect();
        let attended = attention(&q, &keys, &values, None, dim, w.heads);
        let projected = proj_row(&attended, &w.ao_w, &w.ao_b);

        let mut query: Vec<f32> = (0..dim).map(|d| projected[d] + query[d]).collect();
        layer_norm(&mut query, &w.qn_w, &w.qn_b);

        let mut audio_norm = audio;
        for t in 0..FRAME_COUNT {
            layer_norm(&mut audio_norm[t * dim..(t + 1) * dim], &w.an_w, &w.an_b);
        }

        let root = (dim as f32).sqrt();
        let frame = |t: usize| &audio_norm[t * dim..(t + 1) * dim];
        let mut activity = [0.0f32; FRAME_COUNT];
        for (t, value) in activity.iter_mut().enumerate() {
            let row = frame(t);
            *value = dot(row, &query) / root + dot(&w.act_w, row) + w.act_b;
        }
        let mut logits = [0.0f32; OFFSETS_MS.len()];
        for j in 0..OFFSETS_MS.len() {
            let row = frame(14 + j);
            let local = dot(row, &query) / root + dot(&w.bnd_w, row) + w.bnd_b;
            let derivative = activity[14 + j] - activity[14 + j - 1];
            let direction = if is_end == 0 { derivative } else { -derivative };
            logits[j] = local + w.derivative_scale * direction + w.prior[j];
        }

        let probabilities = softmax_scaled(&logits, DECODER_TEMPERATURE);
        probabilities.iter().zip(OFFSETS_MS).map(|(p, o)| p * o).sum()
    }
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
        ensure!(self.schema == SCHEMA, "local head schema {:?} != {SCHEMA:?}", self.schema);
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
        close(self.decoder_temperature, DECODER_TEMPERATURE as f64, "decoder_temperature")?;
        close(self.decoder_amplitude, 1.0, "decoder_amplitude")?;
        close(self.inference_prior_scale, INFERENCE_PRIOR_SCALE, "inference_prior_scale")?;

        let f = &self.feature_contract;
        ensure!(f.center_source == "fixed", "feature center_source must be \"fixed\"");
        ensure!(f.context_ms as usize == CONTEXT_MS, "feature context_ms must be {CONTEXT_MS}");
        ensure!(f.feature_dim == FEATURE_DIM, "feature dim must be {FEATURE_DIM}");
        ensure!(f.frame_count == FRAME_COUNT, "feature frame_count must be {FRAME_COUNT}");
        ensure!(f.frame_ms == FRAME_MS, "feature frame_ms must be {FRAME_MS}");
        ensure!(f.hop_ms == HOP_MS, "feature hop_ms must be {HOP_MS}");
        ensure!(f.n_fft == N_FFT, "feature n_fft must be {N_FFT}");
        ensure!(f.n_mels == N_MELS, "feature n_mels must be {N_MELS}");
        ensure!(f.sample_rate == SAMPLE_RATE, "feature sample_rate must be {SAMPLE_RATE}");
        ensure!(
            f.spectral_distance_bands == SPECTRAL_BANDS,
            "feature spectral_distance_bands must be {SPECTRAL_BANDS}"
        );

        let m = &self.model_config;
        ensure!(m.acoustic_dim == FEATURE_DIM, "model_config.acoustic_dim must be {FEATURE_DIM}");
        ensure!(m.qwen_dim == 1024, "model_config.qwen_dim must be 1024");
        ensure!(m.model_dim == 128, "model_config.model_dim must be 128");
        ensure!(m.heads == 4, "model_config.heads must be 4");
        close(m.prior_sigma_ms, PRIOR_SIGMA_MS, "model_config.prior_sigma_ms")?;
        close(m.activity_weight, 0.25, "model_config.activity_weight")?;
        Ok(())
    }
}

fn close(a: f64, b: f64, what: &str) -> Result<()> {
    ensure!((a - b).abs() <= 1e-6, "{what} {a} != the accepted head's {b}");
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
            ensure!(tensor.dtype() == Dtype::F32, "local head tensor {name} must be F32");
            ensure!(tensor.data().len() == 4, "local head tensor {name} must be scalar");
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

        Ok(Self {
            conv1_w: get("audio.0.weight", &[dim, FEATURE_DIM, 5])?,
            conv1_b: get("audio.0.bias", &[dim])?,
            conv2_w: get("audio.2.weight", &[dim, dim, 5])?,
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

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = out.iter().sum();
    out.iter_mut().for_each(|v| *v /= total);
    out
}

fn softmax_scaled(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = scores.iter().map(|v| v / temperature).collect();
    softmax(&scaled)
}

/// One query against `n` keys, all heads; returns the concatenated head output
/// (before `out_proj`).  `mask[r] == 0.0` removes key `r`.
fn attention(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    mask: Option<&ContextMask>,
    dim: usize,
    heads: usize,
) -> Vec<f32> {
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; dim];
    let n = keys.len();
    for h in 0..heads {
        let q = &query[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; n];
        for r in 0..n {
            let k = &keys[r][h * head_dim..(h + 1) * head_dim];
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
        let p = softmax(&scores);
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for r in 0..n {
                acc += p[r] * values[r][h * head_dim + d];
            }
            out[h * head_dim + d] = acc;
        }
    }
    out
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
        let samples = f32_file(&g.dir.join(g.meta["audio_file"].as_str().expect("audio_file")));
        assert_eq!(samples.len(), g.meta["audio_samples"].as_u64().expect("len") as usize);
        let fixed: Vec<i64> = nums(&g.meta, "fixed_ms").iter().map(|v| *v as i64).collect();
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
            let got = &all[endpoint * FRAME_COUNT * FEATURE_DIM..(endpoint + 1) * FRAME_COUNT * FEATURE_DIM];
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
            eprintln!("endpoint {index:3} ({e}): rust {got:9.4}  python {want:9.4}  d {:.4}", got - want);
        }
        eprintln!("offset max |diff| = {worst:.4} ms");
        assert!(worst < 0.05, "head offsets diverge from the reference: {worst} ms");
    }

    #[test]
    fn refine_matches_python_golden() {
        let Some(g) = golden() else { return };
        let samples = f32_file(&g.dir.join(g.meta["audio_file"].as_str().expect("audio_file")));
        let logits = f32_file(&g.dir.join(g.meta["logits_file"].as_str().expect("logits_file")));
        let hidden = f32_file(&g.dir.join(g.meta["hidden_file"].as_str().expect("hidden_file")));
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
                    eprintln!("word {i:2} {:12} {:>5}: rust {got:.4}  python {want_v:.4}  d {d:.3} ms", item.text, ["start", "end"][key - 1]);
                }
                worst = worst.max(d);
            }
        }
        eprintln!("refine max |diff| = {worst:.4} ms");
        assert!(worst < 0.5, "refined intervals diverge from the reference: {worst} ms");
    }
}
