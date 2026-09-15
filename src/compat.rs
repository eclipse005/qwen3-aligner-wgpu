//! The `qwen-forced-aligner-rs` surface, so this port can replace it.
//!
//! `D:\voxtrans` consumes the CUDA aligner as a library
//! (`D:\qwen-aligner-rs`, crate `qwen-forced-aligner-rs`).  Everything here
//! mirrors that crate's public API — the same type names, the same fields, the
//! same function signatures, the same JSON on disk — so a caller swaps
//! dependencies without touching its source.
//!
//! ```ignore
//! use qwen3_aligner_wgpu::compat::{load_model, AlignRequest, ModelOptions};
//! let model = load_model("models/Qwen3-ForcedAligner-0.6B", ModelOptions::default())?;
//! let result = model.align(AlignRequest::from_paths("a.wav", "a.txt", "Chinese"))?;
//! ```
//!
//! Three places where the two crates **cannot** be identical, each stated rather
//! than papered over:
//!
//! * **`DeviceRequest::Cuda(n)`.**  There is no CUDA here.  It is accepted — the
//!   variant has to exist for a caller's `match` to compile — and resolves to the
//!   best wgpu adapter, which is what the caller meant by "the GPU".  `Cpu` and
//!   `Auto` mean what they say.
//! * **`align` takes `&self`.**  The CUDA crate takes `&self` because its GPU
//!   state is behind `Arc`; ours reuses scratch buffers, so the towers sit behind
//!   a `RefCell`.  Borrowing the same model from two threads at once will panic
//!   rather than serialise; `Qwen3ForcedAligner` is deliberately neither `Send`
//!   nor `Sync` because wgpu's handles are not, and the CUDA crate's
//!   `unsafe impl Send` is a promise this one cannot honestly make.
//! * **`output_ids` is the argmax bucket, not a token id.**  Same as the CUDA
//!   crate's — the name is theirs, the meaning is the timestamp class.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::align_inference::Aligner;
use crate::gpu::DeviceSelector;
use crate::postprocess::AlignItem;

/// One aligned span.  Field-for-field what `qwen-forced-aligner-rs` exports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForcedAlignItem {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// The aligned spans plus the two timestamp streams they came from.
///
/// `output_ids` is the raw argmax per timestamp position (in 80 ms classes, two
/// per word), `raw_timestamp_ms` is that times the segment time, and
/// `fixed_timestamp_ms` is what `_fix_timestamps` made of it.  Keeping all three
/// is what lets a caller see that a given timestamp sits on a near-tie rather
/// than trusting the repaired value blindly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForcedAlignResult {
    pub items: Vec<ForcedAlignItem>,
    pub output_ids: Vec<i64>,
    pub raw_timestamp_ms: Vec<i64>,
    pub fixed_timestamp_ms: Vec<i64>,
}

impl ForcedAlignResult {
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Audio the caller already has, either way the CUDA crate accepts it.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioInput {
    Path(PathBuf),
    /// Mono 16 kHz, in [-1, 1].
    Waveform16Khz(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextInput {
    Path(PathBuf),
    Text(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignRequest {
    pub audio: AudioInput,
    pub text: TextInput,
    pub language: String,
}

impl AlignRequest {
    pub fn new(audio: AudioInput, text: TextInput, language: impl Into<String>) -> Self {
        Self {
            audio,
            text,
            language: language.into(),
        }
    }
    pub fn from_paths(
        audio: impl Into<PathBuf>,
        text: impl Into<PathBuf>,
        language: impl Into<String>,
    ) -> Self {
        Self::new(
            AudioInput::Path(audio.into()),
            TextInput::Path(text.into()),
            language,
        )
    }
}

/// Which engine to load.  `Cuda` exists for source compatibility only — see the
/// module note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    Cpu,
    /// No CUDA backend here; this resolves to the best wgpu adapter.
    Cuda(usize),
    #[default]
    Auto,
}

impl DeviceRequest {
    fn selector(self) -> DeviceSelector {
        match self {
            // The ordinal has no meaning for wgpu, which enumerates its own
            // adapters; the intent — "the GPU" — is what carries over.
            DeviceRequest::Cuda(_) | DeviceRequest::Auto => DeviceSelector::Auto,
            DeviceRequest::Cpu => DeviceSelector::Cpu,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelOptions {
    pub device: DeviceRequest,
}

/// Load the aligner.  Free function for parity with the CUDA crate's
/// `load_model`, which exists so its callers could swap the candle-backed
/// implementation underneath without source changes.  This is the same favour.
pub fn load_model(model_dir: impl AsRef<Path>, options: ModelOptions) -> Result<Qwen3ForcedAligner> {
    Qwen3ForcedAligner::load(model_dir.as_ref(), options)
}

/// Explicit destructor; `drop` spelled out, as in the CUDA crate.
pub fn release_model(model: Qwen3ForcedAligner) {
    drop(model);
}

/// A loaded forced-alignment model.
///
/// Not `Send`/`Sync` on purpose: the towers behind the `RefCell` hold wgpu
/// handles, and the CUDA crate's `unsafe impl Send` is a claim this one cannot
/// back.  A caller that needs to move it between threads has to say so itself.
pub struct Qwen3ForcedAligner {
    inner: RefCell<Aligner>,
}

impl Qwen3ForcedAligner {
    pub fn load(model_dir: &Path, options: ModelOptions) -> Result<Self> {
        Ok(Self {
            inner: RefCell::new(Aligner::load(options.device.selector(), model_dir)?),
        })
    }

    /// The languages the checkpoint supports, lowercased and sorted.
    pub fn supported_languages(&self) -> Vec<String> {
        self.inner.borrow().supported_languages()
    }

    pub fn align(&self, request: AlignRequest) -> Result<ForcedAlignResult> {
        let text = match &request.text {
            TextInput::Path(p) => std::fs::read_to_string(p)
                .with_context(|| format!("read transcript {}", p.display()))?,
            TextInput::Text(t) => t.clone(),
        };
        let (raw_ms, items) = match &request.audio {
            AudioInput::Path(p) => {
                self.inner
                    .borrow_mut()
                    .align_with_raw(p, &text, Some(&request.language))?
            }
            // The waveform is already mono 16 kHz, so it goes straight to the
            // mel — one `borrow_mut` either way, no separate code path.
            AudioInput::Waveform16Khz(w) => {
                self.inner
                    .borrow_mut()
                    .align_samples(w, &text, Some(&request.language))?
            }
        };

        let segment = self.inner.borrow().config().timestamp_segment_time_ms;
        Ok(ForcedAlignResult {
            output_ids: raw_ms.iter().map(|ms| (ms / segment as i64).max(0)).collect(),
            fixed_timestamp_ms: items
                .iter()
                .flat_map(|i| {
                    [
                        (i.start_time * 1000.0).round() as i64,
                        (i.end_time * 1000.0).round() as i64,
                    ]
                })
                .collect(),
            items: items.into_iter().map(Into::into).collect(),
            raw_timestamp_ms: raw_ms,
        })
    }

    /// Align each request in turn.  The CUDA crate's is a loop too; neither
    /// batches into one padded forward.
    pub fn align_batch<I>(&self, requests: I) -> Result<Vec<ForcedAlignResult>>
    where
        I: IntoIterator<Item = AlignRequest>,
    {
        requests
            .into_iter()
            .enumerate()
            .map(|(i, req)| {
                self.align(req)
                    .with_context(|| format!("failed align request {}", i + 1))
            })
            .collect()
    }
}

impl From<AlignItem> for ForcedAlignItem {
    fn from(a: AlignItem) -> Self {
        Self {
            text: a.text,
            start_time: a.start_time,
            end_time: a.end_time,
        }
    }
}

/// Write the items as pretty JSON, creating the parent directory.  Byte-identical
/// to the CUDA crate's writer so downstream files stay comparable.
pub fn write_forced_align_items_json(output: &Path, items: &[ForcedAlignItem]) -> Result<()> {
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(items)?;
    std::fs::write(output, json)?;
    Ok(())
}

/// One manifest line, resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchJob {
    pub request: AlignRequest,
    pub output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ManifestJob {
    audio: PathBuf,
    text: PathBuf,
    output: Option<PathBuf>,
    language: Option<String>,
}

/// Read a JSONL manifest into jobs.
///
/// Paths inside the manifest resolve against the manifest's own directory;
/// `output` resolves against `output_dir`; a missing `language` falls back to
/// `default_language`.  Same rules as the CUDA crate's loader.
pub fn load_manifest_jobs(
    manifest_path: &Path,
    output_dir: &Path,
    default_language: &str,
) -> Result<Vec<BatchJob>> {
    let manifest_text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest_dir = manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut jobs = Vec::new();

    for (line_index, line) in manifest_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let raw: ManifestJob = serde_json::from_str(line).with_context(|| {
            format!(
                "invalid manifest json at {}:{}",
                manifest_path.display(),
                line_index + 1
            )
        })?;
        let audio = resolve_relative_to(manifest_dir, raw.audio);
        let text = resolve_relative_to(manifest_dir, raw.text);
        let output = match raw.output {
            Some(output) => resolve_relative_to(output_dir, output),
            None => default_output_path(&audio, output_dir)?,
        };

        jobs.push(BatchJob {
            request: AlignRequest::new(
                AudioInput::Path(audio),
                TextInput::Path(text),
                raw.language
                    .filter(|language| !language.trim().is_empty())
                    .unwrap_or_else(|| default_language.to_string()),
            ),
            output,
        });
    }

    if jobs.is_empty() {
        anyhow::bail!("manifest has no jobs: {}", manifest_path.display());
    }
    Ok(jobs)
}

fn resolve_relative_to(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn default_output_path(audio: &Path, output_dir: &Path) -> Result<PathBuf> {
    Ok(output_dir.join(format!("{}.json", file_stem_string(audio)?)))
}

fn file_stem_string(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(ToOwned::to_owned)
        .with_context(|| format!("path has no UTF-8 file stem: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `output_ids` is documented as the argmax class, so it must recover the
    /// millisecond stream exactly — that is the only thing tying the two fields
    /// together.
    #[test]
    fn output_ids_and_raw_ms_agree() {
        let segment = 80i64;
        let raw = vec![2000i64, 2080, 2080, 2240];
        let ids: Vec<i64> = raw.iter().map(|ms| (ms / segment).max(0)).collect();
        assert_eq!(ids, vec![25, 26, 26, 28]);
        for (id, ms) in ids.iter().zip(&raw) {
            assert_eq!(id * segment, *ms);
        }
    }

    #[test]
    fn manifest_paths_resolve_against_the_manifest_dir() {
        let base = Path::new(r"D:\proj\manifests");
        assert_eq!(
            resolve_relative_to(base, PathBuf::from("a.wav")),
            Path::new(r"D:\proj\manifests\a.wav")
        );
        let abs = PathBuf::from(r"D:\elsewhere\b.wav");
        assert_eq!(resolve_relative_to(base, abs.clone()), abs);
    }

    #[test]
    fn cuda_device_request_maps_to_the_gpu() {
        assert!(matches!(
            DeviceRequest::Cuda(0).selector(),
            DeviceSelector::Auto
        ));
        assert!(matches!(DeviceRequest::Auto.selector(), DeviceSelector::Auto));
        assert!(matches!(DeviceRequest::Cpu.selector(), DeviceSelector::Cpu));
        assert_eq!(DeviceRequest::default(), DeviceRequest::Auto);
    }

    /// The whole point of this module: a caller's code path — `load_model`,
    /// `AlignRequest`, `align` — has to produce the same items the gate froze.
    ///
    /// Both the path form and the waveform form are exercised, because a caller
    /// that already decoded its audio uses the second.
    #[test]
    fn the_compat_entry_point_matches_the_gold() {
        use crate::gold::{Dtype, GoldJson};

        let dir = crate::gold::model_dir();
        let fixtures = crate::gold::fixtures_dir();
        if !dir.join("config.json").is_file() || !fixtures.is_dir() {
            return;
        }
        let model = load_model(
            &dir,
            ModelOptions {
                device: DeviceRequest::Cpu,
            },
        )
        .expect("load_model");

        assert!(model
            .supported_languages()
            .contains(&"english".to_string()));

        let gold = GoldJson::load(Dtype::Fp32, "15s_en").unwrap();
        let wav = fixtures.join("15s_en.wav");

        // Path form.
        let by_path = model
            .align(AlignRequest::new(
                AudioInput::Path(wav.clone()),
                TextInput::Text(gold.transcript.clone()),
                "English",
            ))
            .expect("align(path)");

        // Waveform form, through the same model value — which is also what
        // proves `align` really is `&self`.
        let samples = crate::mel::load_audio_wav(&wav, 16000).unwrap();
        let by_wave = model
            .align(AlignRequest::new(
                AudioInput::Waveform16Khz(samples),
                TextInput::Text(gold.transcript.clone()),
                "English",
            ))
            .expect("align(waveform)");

        assert_eq!(by_path, by_wave, "path and waveform forms disagree");
        assert_eq!(by_path.items.len(), gold.words.len());
        for (got, want) in by_path.items.iter().zip(&gold.items) {
            assert_eq!(got.text, want.text);
            assert_eq!(got.start_time, want.start_time);
            assert_eq!(got.end_time, want.end_time);
        }
        // The three streams have to line up: two timestamps per word, and
        // `output_ids` is the raw millisecond stream in 80 ms classes.
        assert_eq!(by_path.output_ids.len(), gold.words.len() * 2);
        assert_eq!(by_path.fixed_timestamp_ms.len(), gold.words.len() * 2);
        for (id, ms) in by_path.output_ids.iter().zip(&by_path.raw_timestamp_ms) {
            assert_eq!(id * 80, *ms, "output_ids is not raw_ms in 80 ms classes");
        }
    }
}
