//! qwen3-forced-aligner-wgpu — Qwen3-ForcedAligner in Rust, on wgpu, with a CPU
//! backend alongside it.
//!
//! Forced alignment is the inverse of transcription: audio *and* its transcript
//! go in, and out comes where each word sits.  The model is a single causal pass
//! — no autoregressive loop — so a clip costs one forward, and the alignment is
//! read off a 5000-way classification head at the `<timestamp>` positions.
//!
//! ```no_run
//! use qwen3_aligner_wgpu::align_inference::Aligner;
//! use qwen3_aligner_wgpu::gpu::Backend;
//! # fn main() -> anyhow::Result<()> {
//! let mut aligner = Aligner::load_backend(Backend::parse("auto")?, std::path::Path::new("model"))?;
//! let items = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some("English"))?;
//! # Ok(()) }
//! ```
//!
//! [`align_inference`] is the entry point.  [`words`], [`align_input`] and
//! [`postprocess`] are the three pieces of the output contract: how the
//! transcript is split, how the sequence is assembled, and how the model's
//! timestamps are repaired and paired.
//!
//! [`gpu`], [`mel`] and [`weights`] are device plumbing, the log-mel front end
//! and checkpoint access; [`shaders`], [`decoder`] and the two audio encoders
//! are the model's two towers.
//!
//! Runs against either checkpoint layout — the **`-hf`** export
//! (`model.audio_tower.*` / `score.weight`) or the original one
//! (`thinker.audio_tower.*` / `thinker.lm_head.weight`).  When the checkpoint
//! ships a `<model>/local_head/`, [`local_head`] refines every boundary from the
//! 80 ms class grid to 10 ms.
//! Baselines are in `tools/gold/`; `src/gold.rs` records why the timestamp
//! criterion is a margin rather than a tolerance.

pub mod align_inference;
pub mod align_input;
pub mod audio_encoder;
pub mod audio_encoder_gpu;
pub mod config;
pub mod cpu_decoder;
pub mod cpu_tensor;
pub mod decoder;
pub mod gpu;
pub mod half16;
pub mod load_trace;
pub mod local_head;
pub mod mel;
pub mod mrope;
pub mod paths;
pub mod postprocess;
pub mod shaders;
pub mod weights;
pub mod words;

pub use gpu::{list_devices, Backend, DeviceInfo, DeviceSelector, Gpu};
pub use mel::load_audio_wav;
pub use postprocess::{decode_timestamps, fix_timestamps, AlignItem};
pub use words::split_words;
