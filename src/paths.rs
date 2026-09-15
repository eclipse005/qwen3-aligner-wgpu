//! Where the checkpoint and the test fixtures live.
//!
//! Both default to this machine's layout and both are overridable, so a
//! checkout elsewhere works without editing anything.

use std::path::{Path, PathBuf};

/// The Qwen3-ForcedAligner checkpoint.
///
/// Override with `QALIGN_MODEL`.  The checkpoint is the **`-hf`** one
/// (`architectures = Qwen3ASRForTokenClassification`, tensors under
/// `model.audio_tower.*` / `model.language_model.*` / `score.weight`); the
/// original-layout repository stores different tensor names and will not load.
pub fn model_dir() -> PathBuf {
    env_path("QALIGN_MODEL")
        .unwrap_or_else(|| PathBuf::from(r"D:\Qwen3-ASR\models\Qwen3-ForcedAligner-0.6B-tf"))
}

/// The test fixture wavs: `15s_en`, `30s_zh`, `90s_en`, `90s_ja`, `180s_en`,
/// `180s_zh`.  Override with `QALIGN_FIXTURES`.
pub fn fixtures_dir() -> PathBuf {
    env_path("QALIGN_FIXTURES")
        .unwrap_or_else(|| PathBuf::from(r"D:\qwen3-asr-rs\tests\fixtures"))
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from).filter(|p| p.is_dir())
}

/// A fixture wav by clip name, or `None` when the fixture tree is absent.
pub fn fixture(clip: &str) -> Option<PathBuf> {
    let p = fixtures_dir().join(format!("{clip}.wav"));
    p.is_file().then_some(p)
}

/// The six clips the port was verified against.
pub const CLIPS: [&str; 6] = ["15s_en", "30s_zh", "90s_ja", "90s_en", "180s_en", "180s_zh"];

/// The language each of those clips is in.
pub const LANGUAGES: [&str; 6] = [
    "English", "Chinese", "Japanese", "English", "English", "Chinese",
];

/// The checkpoint's `config.json`, or `None` when it is not on disk.
pub fn config_file() -> Option<PathBuf> {
    let p = model_dir().join("config.json");
    p.is_file().then_some(p)
}

/// True when neither the checkpoint nor the fixtures are present, i.e. this
/// checkout cannot run the model-dependent tests.
pub fn nothing_to_test_against() -> bool {
    config_file().is_none()
}

/// A directory that certainly exists, for tests that write files.
pub fn scratch() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}
