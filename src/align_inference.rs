//! The aligner's forward: audio tower → text prefill → timestamp head.
//!
//! ```text
//! mel ── GpuAudioEncoder::encode ──▶ [n_audio_tokens, 1024]   (ln_post + proj1+gelu+proj2)
//! input_ids ── embed_tokens gather ──▶ [seq, 1024]
//!                                       ↓ scatter the audio rows into the <|audio_pad|> slots
//!                        WgpuTextDecoder::prefill  (28 qwen3 layers, causal, one pass)
//!                                       ↓
//!                          final RMSNorm (all seq rows)
//!                                       ↓
//!                 score GEMV on the <timestamp> rows only  (1024 → 5000)
//!                                       ↓ argmax
//!                              raw_ms = bucket * 80
//! ```
//!
//! Three properties worth knowing before changing anything here:
//!
//! 1. There is no generation loop — one prefill, no KV-cache decode steps.
//! 2. Position encoding is **plain RoPE**: every dimension reads the same
//!    position axis, which is what `section = [half, 0, 0]` below produces.
//! 3. The head is evaluated only on the `<timestamp>` rows.  Scoring the rest
//!    cannot change the argmax of the rows that are read.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use half::f16;
use rayon::prelude::*;

use crate::align_input::AlignerInput;
use crate::postprocess::AlignItem;
use crate::audio_encoder_gpu::GpuAudioEncoder;
use crate::config::AsrConfig;
use crate::cpu_tensor::{CpuTensor, CpuWeight};
use crate::decoder::{TextConfig, WgpuTextDecoder};
use crate::gpu::{DeviceSelector, Gpu};
use crate::mrope::{compute_mrope_cos_sin, text_positions};
use crate::weights::{self, RawTensor};

/// Everything the aligner needs from `config.json`.
pub struct AlignerConfig {
    pub audio_cfg: crate::config::AudioEncoderConfig,
    pub text_cfg: TextConfig,
    pub timestamp_token_id: u32,
    pub timestamp_segment_time_ms: f64,
    pub num_labels: usize,
    pub audio_token_id: u32,
    pub max_seq: usize,
    /// The checkpoint's declared language list.
    pub support_languages: Vec<String>,
}

impl AlignerConfig {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let cfg = AsrConfig::from_file(&dir.join("config.json"))?;
        let head = cfg
            .align
            .as_ref()
            .context("checkpoint has no forced-aligner head (timestamp_token_id missing)")?;
        Ok(Self {
            audio_cfg: cfg.thinker_config.audio_config.clone(),
            text_cfg: TextConfig::from_model_dir(dir)?,
            timestamp_token_id: head.timestamp_token_id as u32,
            timestamp_segment_time_ms: head.timestamp_segment_time_ms,
            num_labels: head.num_labels,
            audio_token_id: cfg.thinker_config.audio_token_id as u32,
            max_seq: 8192,
            support_languages: cfg.support_languages.clone(),
        })
    }
}

/// The two towers, in whichever form the caller asked for.
///
/// Only the towers differ between devices.  Everything between them — the
/// embedding gather, the audio scatter, the final norm, the timestamp head — is
/// host code and is shared verbatim, which is why this enum has exactly two
/// methods and no `align` of its own.
enum Backend {
    /// `Gpu` is not cloneable and the decoder owns it; the tower borrows.
    Gpu {
        encoder: GpuAudioEncoder,
        decoder: WgpuTextDecoder,
    },
    Cpu {
        encoder: crate::audio_encoder::CpuAudioEncoder,
        decoder: crate::cpu_decoder::CpuTextDecoder,
    },
}

impl Backend {
    /// Mel in, projected audio frames out — `[n_audio_tokens, hidden]` f16.
    ///
    /// The CPU tower computes in f32 (`CpuAudioEncoder::forward ->
    /// Vec<f32>`), while the text tower on **both** paths takes f16 bytes.  The
    /// widening is therefore undone here, at the boundary, so the two devices
    /// hand the text tower the same thing.  Rounding at a different point would
    /// be a numerical difference at the entrance to the shared half of the graph.
    fn audio_embeds(&mut self, mel: &[f32], valid_frames: usize) -> Result<Vec<f16>> {
        match self {
            Backend::Gpu { encoder, decoder } => {
                encoder.encode(decoder.gpu(), mel, 128, valid_frames)
            }
            Backend::Cpu { encoder, .. } => {
                let f = encoder.forward(mel, 128, valid_frames)?;
                Ok(f.into_iter().map(f16::from_f32).collect())
            }
        }
    }

    /// Make sure the KV cache holds `need` positions — a no-op when it already
    /// does, which is the common case (the capacity is grow-only).
    fn ensure_capacity(&mut self, need: usize) -> usize {
        match self {
            Backend::Gpu { decoder, .. } => decoder.ensure_capacity(need),
            Backend::Cpu { decoder, .. } => decoder.ensure_capacity(need),
        }
    }

    /// `[seq, hidden]` f16 hidden states after the last text layer.
    fn text_hidden(&mut self, bytes: &[u8], seq: usize) -> Result<Vec<f16>> {
        self.ensure_capacity(seq);
        match self {
            Backend::Gpu { decoder, .. } => {
                // The ASR path's first decode token is meaningless here; the
                // layer outputs are what we need and `prefill` exposes them.
                let _ = decoder.prefill(bytes, seq, 0)?;
                let buf = decoder
                    .debug_prefill_h
                    .as_ref()
                    .context("prefill did not expose its hidden states")?;
                decoder.read_f16(buf, seq * decoder.cfg.hidden_size)
            }
            Backend::Cpu { decoder, .. } => decoder.prefill_hidden(bytes, seq, 0),
        }
    }

    fn describe(&self) -> String {
        match self {
            Backend::Gpu { decoder, .. } => decoder.gpu().describe(),
            Backend::Cpu { decoder, .. } => decoder.describe(),
        }
    }
}

pub struct Aligner {
    backend: Backend,
    /// Built once: the BPE tokenizer is a 300 MB vocabulary and re-reading it per
    /// clip was the CLI's job only because this used to live there.
    input_builder: crate::align_input::InputBuilder,
    /// `[vocab, hidden]`, kept on the host: the gather is `n_text_tokens` rows out
    /// of 152 064, so uploading the table to do it would move 300 MB to read back 1.
    embed_tokens: RawTensor,
    /// `[num_labels, hidden]`, host-side for the first cut — see the module note.
    score: CpuWeight,
    final_norm: Vec<f32>,
    cfg: AlignerConfig,
    pub timings: Timings,
}

/// One row per phase, so no time falls between the buckets.
///
/// `total_ms` is the model forward only (`enc .. head`); everything the caller
/// pays outside it has its own field, and the CLI prints them all.  A residual
/// with no name is where optimisations go to hide, so there is none.
#[derive(Debug, Default, Clone, Copy)]
pub struct Timings {
    /// `load_audio_wav`: read, PCM decode, downmix, and resample if the source
    /// is not already 16 kHz.
    pub wav_ms: f64,
    /// `mel_features`: the log-mel STFT.
    pub mel_ms: f64,
    /// `split_words`: the transcript's word segmentation (nagisa for Japanese).
    pub words_ms: f64,
    /// `InputBuilder::build`: tokenise and assemble the model's input sequence.
    pub build_ms: f64,
    pub enc_ms: f64,
    pub gather_ms: f64,
    pub prefill_ms: f64,
    pub head_ms: f64,
    /// `decode_timestamps`: repair the raw buckets and pair them with the words.
    pub post_ms: f64,
    pub total_ms: f64,
}

impl Aligner {
    pub fn load(selector: DeviceSelector, model_dir: &Path) -> Result<Self> {
        let t = std::time::Instant::now();
        let cfg = AlignerConfig::from_model_dir(model_dir)?;
        crate::load_trace::note("config", t);
        // The tokenizer's BPE is a ~270 ms single-threaded JSON parse with nothing
        // to do with the device, so it parses on its own thread and is joined at
        // the end — by then it is always ready, and it is off the load's critical
        // path instead of being 9% of it.
        //
        // The Japanese tagger's embedded model rides along on the same thread,
        // for the same reason and with a bigger number attached: it is ~451 ms,
        // and `split_japanese` would otherwise spend it inside the first Japanese
        // clip's word-splitting phase.
        let tok_dir = model_dir.to_path_buf();
        let tok_ts = cfg.timestamp_token_id;
        let tok_thread = std::thread::spawn(move || {
            crate::words::warm_japanese_tagger();
            crate::align_input::InputBuilder::load(&tok_dir, tok_ts)
        });
        let t = std::time::Instant::now();
        let weights: HashMap<String, RawTensor> = weights::load_tensors(model_dir)?;
        crate::load_trace::note("safetensors (mmap + header)", t);

        // Plain RoPE: all `head_dim/2` frequencies on axis 0, which is `0..n`.
        // The `-hf` checkpoint declares no `mrope_section` (asserted in
        // `config::tests`), so flattening the axis map back to plain RoPE is what
        // matches it.  Both backends need the same table.
        let t = std::time::Instant::now();
        let half = cfg.text_cfg.head_dim / 2;
        let (cos, sin) = compute_mrope_cos_sin(
            &text_positions(cfg.max_seq),
            cfg.text_cfg.head_dim,
            1_000_000.0,
            &[half, 0, 0],
            false,
        );
        let cos: Vec<f16> = cos.iter().map(|&v| f16::from_f32(v)).collect();
        let sin: Vec<f16> = sin.iter().map(|&v| f16::from_f32(v)).collect();
        crate::load_trace::note("rope tables", t);

        let backend = if matches!(selector, DeviceSelector::Cpu) {
            let mut decoder = crate::cpu_decoder::CpuTextDecoder::load(
                model_dir,
                "thinker.model",
                cfg.text_cfg.clone(),
                cfg.max_seq,
                cfg.max_seq,
            )?;
            decoder.set_rope_tables(&cos, &sin);
            let encoder = crate::audio_encoder::CpuAudioEncoder::load(
                &weights,
                "thinker.audio_tower",
                &cfg.audio_cfg,
            )?;
            Backend::Cpu { encoder, decoder }
        } else {
            let t = std::time::Instant::now();
            let gpu = pollster::block_on(Gpu::new_with(selector.clone()))
                .with_context(|| format!("open device {selector:?}"))?;
            crate::load_trace::note("adapter + device", t);
            let t = std::time::Instant::now();
            crate::load_trace::transfer::reset();
            let decoder = WgpuTextDecoder::load(
                gpu,
                model_dir,
                "thinker.model",
                cfg.text_cfg.clone(),
                cfg.max_seq,
                cfg.max_seq,
            )?;
            crate::load_trace::note("text decoder (incl. pipelines)", t);
            crate::load_trace::transfer::note("text decoder: transfer");
            decoder.set_rope_tables(&cos, &sin);
            let t = std::time::Instant::now();
            crate::load_trace::transfer::reset();
            let encoder = GpuAudioEncoder::load(
                decoder.gpu(),
                &weights,
                "thinker.audio_tower",
                &cfg.audio_cfg,
                cfg.audio_cfg.n_window_infer,
            )?;
            crate::load_trace::note("gpu audio tower", t);
            crate::load_trace::transfer::note("gpu audio tower: transfer");
            Backend::Gpu { encoder, decoder }
        };

        let embed_tokens = weights
            .get("thinker.model.embed_tokens.weight")
            .context("embed_tokens missing")?
            .clone();
        let score = CpuWeight {
            data: weights
                .get("score.weight")
                .context("score.weight missing (is this the -hf checkpoint?)")?
                .to_f32_vec()?,
            rows: cfg.num_labels,
            cols: cfg.text_cfg.hidden_size,
        };
        if score.data.len() != score.rows * score.cols {
            bail!(
                "score.weight has {} values, expected {}x{}",
                score.data.len(),
                score.rows,
                score.cols
            );
        }
        let final_norm = weights
            .get("thinker.model.norm.weight")
            .context("final norm weight missing")?
            .to_f32_vec()?;

        let t = std::time::Instant::now();
        let input_builder = tok_thread
            .join()
            .map_err(|_| anyhow::anyhow!("tokenizer thread panicked"))??;
        crate::load_trace::note("input builder join (tokenizer)", t);

        Ok(Self {
            backend,
            input_builder,
            embed_tokens,
            score,
            final_norm,
            cfg,
            timings: Timings::default(),
        })
    }

    /// Which device the towers are running on.
    pub fn backend_name(&self) -> &'static str {
        match &self.backend {
            Backend::Gpu { .. } => "gpu",
            Backend::Cpu { .. } => "cpu",
        }
    }

    pub fn describe(&self) -> String {
        self.backend.describe()
    }

    pub fn config(&self) -> &AlignerConfig {
        &self.cfg
    }

    /// The reference's entry point: an audio file, its transcript, a language —
    /// back come the aligned words with times in seconds.
    ///
    /// This is `Qwen3ForcedAligner.align(audio, text, language)`.  Everything the
    /// caller used to have to do by hand — decode and resample the wav, run the
    /// log-mel, split the transcript into words, assemble and tokenise
    /// `[151669] + [151676]xN + [151670] + words`, repair the timestamps, pair
    /// them with the words — happens in here, in the reference's order.
    pub fn align(
        &mut self,
        audio: &Path,
        text: &str,
        language: Option<&str>,
    ) -> Result<Vec<AlignItem>> {
        Ok(self.align_with_raw(audio, text, language)?.1)
    }

    /// As [`Self::align`], but also hands back the pre-repair millisecond stream.
    ///
    /// The gate needs both axes: the raw argmax is where `margin[i]` is
    /// meaningful (it describes the reference's choice at position `i`), while
    /// the repaired values are what a caller sees.  Comparing the *repaired*
    /// stream against a raw margin is an index mismatch — the repair can move a
    /// value in from elsewhere — and it reports failures that are not there.
    pub fn align_with_raw(
        &mut self,
        audio: &Path,
        text: &str,
        language: Option<&str>,
    ) -> Result<(Vec<i64>, Vec<AlignItem>)> {
        let t = std::time::Instant::now();
        let samples = crate::mel::load_audio_wav(audio, 16000)?;
        self.timings.wav_ms = t.elapsed().as_secs_f64() * 1000.0;
        self.align_samples(&samples, text, language)
    }

    /// As [`Self::align_with_raw`], for audio the caller already decoded:
    /// mono 16 kHz, in [-1, 1].
    pub fn align_samples(
        &mut self,
        samples: &[f32],
        text: &str,
        language: Option<&str>,
    ) -> Result<(Vec<i64>, Vec<AlignItem>)> {
        self.check_language(language)?;
        let t = std::time::Instant::now();
        let (mel, _bins, _frames) = crate::mel::mel_features(samples)?;
        self.timings.mel_ms = t.elapsed().as_secs_f64() * 1000.0;
        let valid = crate::align_input::valid_mel_frames(samples.len());

        // The processor right-pads the mel axis to a multiple of `n_window * 2`
        // with 0.0 before the encoder ever sees it.
        let padded = crate::align_input::padded_mel_frames(valid, self.input_builder.n_window());
        let mut mel = mel;
        mel.resize(self.cfg.audio_cfg.num_mel_bins * padded, 0.0);

        let t = std::time::Instant::now();
        let words = crate::words::split_words(text, language)?;
        self.timings.words_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let input = self.input_builder.build(&words, valid)?;
        self.timings.build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let raw_ms = self.align_raw_ms(&mel, valid, &input)?;
        let t = std::time::Instant::now();
        let items = crate::postprocess::decode_timestamps(&words, &raw_ms)?;
        self.timings.post_ms = t.elapsed().as_secs_f64() * 1000.0;
        Ok((raw_ms, items))
    }

    /// The languages the forced aligner supports — the same set as
    /// `FORCED_ALIGNER_LANGUAGES` in `processing_qwen3_asr.py`, which is where the
    /// reference actually enforces it (`prepare_forced_aligner_inputs` raises on
    /// anything else).
    ///
    /// Read from the checkpoint when its config declares `support_languages`, as
    /// the original-layout checkpoint does.  The **`-hf` repackaging dropped that
    /// field**, so for the checkpoint this port is gated against the list has to
    /// come from here; returning an empty list would silently accept every
    /// language and defer the failure to a wrong alignment.
    pub const FORCED_ALIGNER_LANGUAGES: [&'static str; 11] = [
        "Chinese", "Cantonese", "English", "French", "German", "Italian",
        "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
    ];

    /// The reference's `get_supported_languages()`: lowercased and sorted.
    pub fn supported_languages(&self) -> Vec<String> {
        let declared = if self.cfg.support_languages.is_empty() {
            Self::FORCED_ALIGNER_LANGUAGES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        } else {
            self.cfg.support_languages.clone()
        };
        let mut v: Vec<String> = declared.iter().map(|s| s.to_lowercase()).collect();
        v.sort();
        v.dedup();
        v
    }

    /// The reference rejects a language outside the set rather than aligning
    /// against the wrong tokeniser.
    fn check_language(&self, language: Option<&str>) -> Result<()> {
        let Some(l) = language else { return Ok(()) };
        let l = l.to_lowercase();
        let supported = self.supported_languages();
        anyhow::ensure!(
            supported.contains(&l),
            "language {l:?} is not supported by the forced aligner; supported: {supported:?}"
        );
        Ok(())
    }

    /// One sample: mel + a built input sequence in, one raw millisecond value per
    /// timestamp token out (`2 * words` of them, in order).
    ///
    /// The primitive `align` is built on, for a caller driving the stages itself
    /// — the gate does, so that a divergence can be attributed to a phase.  It
    /// takes the mel and the assembled input sequence rather than a file, and
    /// returns the pre-repair milliseconds.
    pub fn align_raw_ms(&mut self, mel: &[f32], valid_frames: usize, input: &AlignerInput) -> Result<Vec<i64>> {
        let t_all = std::time::Instant::now();
        let hs = self.cfg.text_cfg.hidden_size;
        let seq = input.seq_len();

        // ---- audio tower ----
        let t = std::time::Instant::now();
        let audio = self.backend.audio_embeds(mel, valid_frames)?;
        self.timings.enc_ms = t.elapsed().as_secs_f64() * 1000.0;
        if audio.len() != input.n_audio_tokens * self.cfg.audio_cfg.output_dim {
            bail!(
                "audio tower returned {} values, expected {}x{}",
                audio.len(),
                input.n_audio_tokens,
                self.cfg.audio_cfg.output_dim
            );
        }
        if self.cfg.audio_cfg.output_dim != hs {
            bail!(
                "projector output {} != text hidden {hs}; the scatter needs them equal",
                self.cfg.audio_cfg.output_dim
            );
        }

        // ---- text embeddings + scatter (host) ----
        let t = std::time::Instant::now();
        let mut hidden = vec![f16::ZERO; seq * hs];
        let mut audio_cursor = 0usize;
        {
            let mut row = Vec::with_capacity(hs * 2);
            for (pos, &id) in input.input_ids.iter().enumerate() {
                let dst = &mut hidden[pos * hs..(pos + 1) * hs];
                if id == self.cfg.audio_token_id {
                    let src = &audio[audio_cursor * hs..(audio_cursor + 1) * hs];
                    dst.copy_from_slice(src);
                    audio_cursor += 1;
                } else {
                    row.clear();
                    self.embed_tokens.append_f16_row_le(id as usize, hs, &mut row)?;
                    for (i, c) in row.chunks_exact(2).enumerate() {
                        dst[i] = f16::from_ne_bytes([c[0], c[1]]);
                    }
                }
            }
        }
        if audio_cursor != input.n_audio_tokens {
            bail!(
                "scattered {audio_cursor} audio rows but the sequence has {} audio tokens",
                input.n_audio_tokens
            );
        }
        self.timings.gather_ms = t.elapsed().as_secs_f64() * 1000.0;

        // ---- 28 text layers, one pass ----
        let t = std::time::Instant::now();
        let mut bytes = Vec::with_capacity(hidden.len() * 2);
        for v in &hidden {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        drop(hidden);
        let h = self.backend.text_hidden(&bytes, seq)?;
        self.timings.prefill_ms = t.elapsed().as_secs_f64() * 1000.0;
        drop(bytes);

        // ---- final norm + timestamp head (host) ----
        let t = std::time::Instant::now();
        let rows: Vec<usize> = input.timestamp_positions.clone();
        let normed: Vec<f32> = rows
            .par_iter()
            .flat_map_iter(|&r| {
                let row = &h[r * hs..(r + 1) * hs];
                let mut acc = 0.0f32;
                for (x, w) in row.iter().zip(&self.final_norm) {
                    let xf = x.to_f32() * w;
                    acc += xf * xf;
                }
                let inv = 1.0 / (acc / hs as f32 + self.cfg.text_cfg.rms_norm_eps).sqrt();
                let out: Vec<f32> = row
                    .iter()
                    .zip(&self.final_norm)
                    .map(|(x, w)| x.to_f32() * w * inv)
                    .collect();
                out.into_iter()
            })
            .collect();

        let n_labels = self.cfg.num_labels;
        // One batched GEMM instead of a hand-rolled dot-product loop: the loop
        // measured 29 GFLOPS on 180s_zh (266 ms), which is an order of magnitude
        // below what the `gemm` crate already reaches in `cpu_tensor::linear`.
        let normed_t = CpuTensor::new(normed, vec![rows.len(), hs]);
        let logits = crate::cpu_tensor::linear(&normed_t, &self.score);
        let raw_ms: Vec<i64> = logits
            .data
            .par_chunks_exact(n_labels)
            .map(|row| {
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for (l, &v) in row.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = l;
                    }
                }
                (best as f64 * self.cfg.timestamp_segment_time_ms) as i64
            })
            .collect();
        self.timings.head_ms = t.elapsed().as_secs_f64() * 1000.0;
        self.timings.total_ms = t_all.elapsed().as_secs_f64() * 1000.0;
        Ok(raw_ms)
    }
}
