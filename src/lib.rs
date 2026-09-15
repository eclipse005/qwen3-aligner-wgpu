//! qwen3-forced-aligner-wgpu — Qwen3-ForcedAligner in Rust, on wgpu, with a CPU
//! backend alongside it.
//!
//! Forced alignment is the inverse of transcription: audio *and* its transcript
//! go in, and out comes where each word sits.  The model is a single causal pass
//! — no autoregressive loop — so a clip costs one forward, and the alignment is
//! read off a 5000-way classification head at the positions carrying the
//! `<timestamp>` marker.
//!
//! # Using it
//!
//! ```no_run
//! use qwen3_aligner_wgpu::align_inference::Aligner;
//! use qwen3_aligner_wgpu::gpu::DeviceSelector;
//! # fn main() -> anyhow::Result<()> {
//! let mut aligner = Aligner::load(DeviceSelector::parse("auto")?, std::path::Path::new("model"))?;
//! let items = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some("English"))?;
//! for it in items {
//!     println!("{}\t{:.3}\t{:.3}", it.text, it.start_time, it.end_time);
//! }
//! # Ok(()) }
//! ```
//!
//! [`align_inference`] is the entry point; [`words`], [`align_input`] and
//! [`postprocess`] are the three pieces of the contract that decide what the
//! output *is* — how the transcript is split into words, how the sequence is
//! assembled, and how the model's timestamps are repaired and paired.
//!
//! # What came from where
//!
//! The infrastructure is lifted from the sibling ASR port (`D:\qwen3-asr-wgpu`,
//! finished and verified there at 12/12) because it is model-independent:
//! [`gpu`] (device selection, the persisted pipeline cache, staged uploads),
//! [`mel`] (the torch-compatible log-mel front end and wav loading, with the
//! vendored soxr HQ resampler), [`weights`] (safetensors/mmapped access in the
//! f16 word layout), and the shaders, text decoder and audio towers that go with
//! them.  [`config`], [`cpu_tensor`] and [`mrope`] are the small pieces the
//! reference paths need.
//!
//! # The reference it is gated against
//!
//! `transformers`' native `Qwen3ASRForTokenClassification`, run on the `-hf`
//! checkpoint — **not** the original-layout one, which stores different tensor
//! names.  The frozen baselines live in `tools/gold/`, and `docs/baseline-gold.md`
//! records how they were produced, why the timestamp criterion is a *margin* and
//! not a tolerance, and what is still unverified.

pub mod align_inference;
pub mod align_input;
pub mod audio_encoder;
pub mod audio_encoder_gpu;
pub mod config;
pub mod cpu_decoder;
pub mod cpu_tensor;
pub mod decoder;
pub mod gold;
pub mod gpu;
pub mod mel;
pub mod mrope;
pub mod postprocess;
pub mod shaders;
pub mod weights;
pub mod words;

pub use gold::{Dtype, GoldJson, Verdict};
pub use gpu::{list_devices, DeviceInfo, DeviceSelector, Gpu};
pub use mel::load_audio_wav;
pub use postprocess::{decode_timestamps, fix_timestamps, AlignItem};
pub use words::split_words;
