//! qwen3-forced-aligner-wgpu — a wgpu port of Qwen3-ForcedAligner.
//!
//! **Status: skeleton.**  The infrastructure below was lifted verbatim from the
//! sibling ASR port (`D:\qwen3-asr-wgpu`, which is finished and verified) because
//! it is model-independent:
//!
//! * [`gpu`] — device selection (runtime-first), the persisted pipeline cache,
//!   staged uploads, buffer plumbing;
//! * [`mel`] — the log-mel front end (torch-compatible STFT) + wav loading with
//!   the vendored soxr HQ resampler;
//! * [`weights`] — safetensors/mmapped weight access in the f16 word layout;
//! * [`mrope`] — the MRoPE tables, if the aligner's text tower uses them;
//! * [`config`] / [`cpu_tensor`] — config parsing and the small CPU tensor used
//!   by the reference paths.
//!
//! What is *not* here yet — and is the actual porting work — is the aligner
//! itself: its input construction (audio **plus text**), its model forward, and
//! the token→word→timestamp decode.  See `HANDOFF.md` for the exact upstream
//! files those come from and how the gate works.

pub mod config;
pub mod cpu_tensor;
pub mod gpu;
pub mod mel;
pub mod mrope;
pub mod weights;

pub use gpu::{list_devices, DeviceInfo, DeviceSelector, Gpu};
pub use mel::load_audio_wav;
