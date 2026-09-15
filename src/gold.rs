//! The frozen baseline: loading `tools/gold/` and comparing a run against it.
//!
//! The gold generator (`D:\Qwen3-ASR\run_hf_aligner.py`) writes, per clip, a
//! `.tsv` (`word <TAB> start <TAB> end`) and a `.json` carrying the provenance
//! plus the intermediates a port needs to debug with: the word list, the raw
//! masked argmax buckets (`raw_masked_ms`, in milliseconds) and the repaired
//! values (`fixed_ms`).
//!
//! Gold lives under a **dtype subdirectory** — `f16/`, `bf16/`, `fp32/` — because
//! the reference is dtype-dependent: it is whatever `torch_dtype` the caller
//! passes, and the 5000-way argmax moves by one 80 ms bucket on a handful of
//! words between them.  A port picks the directory matching the precision it
//! computes in (`--dtype auto` on a bf16-capable device, `f16` otherwise).
//!
//! What is **not** dtype-dependent, and is therefore the hard gate: the word
//! sequence, the line count, ordering, and the fact that both endpoints of every
//! word lie inside the audio.  The tokeniser fixes the words and the model emits
//! exactly two timestamps per word, so no numerical choice can change them.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::postprocess::AlignItem;

/// One timestamp bucket, in milliseconds.  Also in `config.json`.
pub const TIMESTAMP_SEGMENT_MS: i64 = 80;

/// Below this top-1/top-2 logit gap the reference's own answer is not stable.
///
/// Measured, not chosen: every divergence this port has ever produced sits at a
/// position with a margin at or under 0.00781, and the reference disagrees with
/// *itself* at those positions — `180s_en` index 253 is chosen differently by
/// CUDA-fp16 than by CUDA-fp32, CPU-fp32 and CPU-fp16 (whose margin there is
/// exactly 0.00000).  The next-lowest margins in that clip are 0.0098 and up, so
/// 0.01 separates "the model decided" from "rounding decided".
pub const MARGIN_NOISE_FLOOR: f32 = 0.01;

/// How a run's timestamp stream compares, split by whether the reference was
/// actually decided at the positions that moved.
#[derive(Debug, Clone, Default)]
pub struct TsVerdict {
    /// Endpoints that differ at all.
    pub moved: usize,
    /// Largest absolute difference, in milliseconds.
    pub max_delta_ms: i64,
    /// Deviations at positions the reference was confident about.  **Any of
    /// these fails the gate** — they are real disagreements, not noise.
    pub beyond_floor: usize,
    /// Smallest margin among the `beyond_floor` positions.
    pub worst_confident_margin: f32,
    /// Deviations excused by [`MARGIN_NOISE_FLOOR`], with the smallest margin
    /// seen among them — recorded so the excused set cannot quietly grow.
    pub excused: usize,
    pub worst_excused_margin: f32,
}

impl TsVerdict {
    /// `want` and `margin` are the frozen reference's; `margin` may be empty if
    /// the gold predates it, in which case nothing is excused.
    pub fn of(got: &[i64], want: &[i64], margin: &[f32]) -> Self {
        let mut v = TsVerdict {
            worst_confident_margin: f32::INFINITY,
            worst_excused_margin: 0.0,
            ..Default::default()
        };
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let d = (a - b).abs();
            if d == 0 {
                continue;
            }
            v.moved += 1;
            v.max_delta_ms = v.max_delta_ms.max(d);
            let m = margin.get(i).copied().unwrap_or(f32::INFINITY);
            if m > MARGIN_NOISE_FLOOR {
                v.beyond_floor += 1;
                v.worst_confident_margin = v.worst_confident_margin.min(m);
            } else {
                v.excused += 1;
                v.worst_excused_margin = v.worst_excused_margin.max(m);
            }
        }
        if v.beyond_floor == 0 {
            v.worst_confident_margin = f32::NAN;
        }
        v
    }

    /// The hard part of the timestamp gate.
    pub fn passes(&self) -> bool {
        self.beyond_floor == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F16,
    Bf16,
    Fp32,
}

impl Dtype {
    pub fn dir_name(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
            Dtype::Fp32 => "fp32",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "float16" | "half" => Some(Dtype::F16),
            "bf16" | "bfloat16" => Some(Dtype::Bf16),
            "fp32" | "float32" | "f32" => Some(Dtype::Fp32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldJson {
    pub clip: String,
    pub language: String,
    pub transcript: String,
    pub words: Vec<String>,
    pub word_count: usize,
    pub seq_len: usize,
    pub n_audio_tokens: usize,
    pub timestamp_token_id: i64,
    pub timestamp_segment_time_ms: f64,
    pub raw_masked_ms: Vec<i64>,
    pub fixed_ms: Vec<i64>,
    pub audio_seconds: f64,
    pub dtype: String,
    pub model_sha256: String,
    #[serde(default)]
    pub input_ids: Vec<u32>,
    #[serde(default)]
    pub n_timestamp_tokens: usize,
    #[serde(default)]
    pub mel_frames: usize,
    /// The reference's own top-1 minus top-2 logit at each timestamp position.
    /// Absent from gold frozen before the margin gate existed.
    #[serde(default)]
    pub margin: Vec<f32>,
    #[serde(default)]
    pub items: Vec<GoldItem>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldItem {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Locate `tools/gold` inside this crate.
pub fn gold_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools").join("gold")
}

/// Locate the fixture wavs.  The ASR round owns them; override with
/// `QALIGN_FIXTURES` when they move.
pub fn fixtures_dir() -> PathBuf {
    std::env::var_os("QALIGN_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\qwen3-asr-rs\tests\fixtures"))
}

/// Locate the `-hf` checkpoint.  Override with `QALIGN_MODEL`.
///
/// Note this is `Qwen/Qwen3-ForcedAligner-0.6B-hf` on ModelScope — the *native*
/// layout (`model.audio_tower.*` / `model.language_model.*` / `score.weight`),
/// not the original `thinker.*` one.
pub fn model_dir() -> PathBuf {
    std::env::var_os("QALIGN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\Qwen3-ASR\models\Qwen3-ForcedAligner-0.6B-tf"))
}

/// The clips the gate covers, in the order the ASR round froze them.
pub const CLIPS: [&str; 6] = [
    "15s_en", "30s_zh", "90s_ja", "90s_en", "180s_en", "180s_zh",
];

pub const LANGUAGES: [&str; 6] = [
    "English", "Chinese", "Japanese", "English", "English", "Chinese",
];

impl GoldJson {
    pub fn load(dtype: Dtype, clip: &str) -> anyhow::Result<Self> {
        let path = gold_root().join(dtype.dir_name()).join(format!("{clip}.json"));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// The reference TSV body, reconstructed from the frozen items.
    pub fn tsv(&self) -> String {
        let mut out = String::new();
        for it in &self.items {
            out.push_str(&format!("{}\t{:.3}\t{:.3}\n", it.text, it.start_time, it.end_time));
        }
        out
    }

    /// Expected `(start, end)` in whole milliseconds — the lossless form, since
    /// the TSV's 3 decimals are exactly the millisecond resolution.
    pub fn expected_ms(&self) -> Vec<(i64, i64)> {
        self.fixed_ms
            .chunks_exact(2)
            .map(|c| (c[0], c[1]))
            .collect()
    }
}

/// How one run compared against one gold.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub clip: String,
    pub dtype: Dtype,
    /// Hard gate: the word sequence and the number of words.
    pub words_match: bool,
    pub first_word_diff: Option<(usize, String, String)>,
    /// Largest absolute per-endpoint difference, in milliseconds.
    pub max_delta_ms: i64,
    /// How many of the `2 * n` endpoints moved by more than one bucket.
    pub beyond_one_bucket: usize,
    /// How many words changed only inside the one-bucket tolerance.
    pub shifted_words: usize,
    /// Words whose repaired interval is empty or runs backwards.
    pub malformed: usize,
    /// Words that fall outside the audio at all.
    pub out_of_range: usize,
    pub audio_seconds: f64,
    pub elapsed_s: f64,
}

impl Verdict {
    /// The gate: words verbatim, nothing malformed, and every timestamp within
    /// `bucket_tolerance` 80 ms buckets of the reference.
    pub fn passes(&self, bucket_tolerance: i64) -> bool {
        self.words_match
            && self.malformed == 0
            && self.out_of_range == 0
            && self.max_delta_ms <= bucket_tolerance * TIMESTAMP_SEGMENT_MS
    }

    pub fn summary(&self) -> String {
        let gate = if self.passes(1) { "PASS" } else { "FAIL" };
        let mut s = format!(
            "{:<8} {:<4} {:<6} words-ok={:<5} maxdelta={:>4}ms  shifted={:<3} \
             beyond-1bucket={:<3} malformed={} out-of-range={}  {:.2}s",
            self.clip,
            self.dtype.dir_name(),
            gate,
            self.words_match,
            self.max_delta_ms,
            self.shifted_words,
            self.beyond_one_bucket,
            self.malformed,
            self.out_of_range,
            self.elapsed_s,
        );
        if let Some((i, want, got)) = &self.first_word_diff {
            s.push_str(&format!("\n    first word diff at {i}: gold={want:?} got={got:?}"));
        }
        s
    }
}

/// Compare a run's items against the gold.
pub fn compare(
    gold: &GoldJson,
    dtype: Dtype,
    got: &[AlignItem],
    elapsed_s: f64,
) -> Verdict {
    let mut first_word_diff = None;
    let mut words_match = got.len() == gold.words.len();
    for (i, w) in gold.words.iter().enumerate() {
        match got.get(i) {
            Some(g) if &g.text == w => {}
            other => {
                words_match = false;
                if first_word_diff.is_none() {
                    first_word_diff = Some((
                        i,
                        w.clone(),
                        other.map(|o| o.text.clone()).unwrap_or_else(|| "<missing>".into()),
                    ));
                }
            }
        }
    }

    let expected = gold.expected_ms();
    let mut max_delta_ms = 0i64;
    let mut beyond_one_bucket = 0usize;
    let mut shifted = 0usize;
    let mut malformed = 0usize;
    let mut out_of_range = 0usize;

    for (i, item) in got.iter().enumerate() {
        let start = (item.start_time * 1000.0).round() as i64;
        let end = (item.end_time * 1000.0).round() as i64;
        if end < start {
            malformed += 1;
        }
        if item.start_time < 0.0 || item.end_time > gold.audio_seconds + 1.0 {
            out_of_range += 1;
        }
        if let Some(&(es, ee)) = expected.get(i) {
            let ds = (start - es).abs();
            let de = (end - ee).abs();
            let worst = ds.max(de);
            max_delta_ms = max_delta_ms.max(worst);
            if worst > 0 {
                shifted += 1;
            }
            if worst > TIMESTAMP_SEGMENT_MS {
                beyond_one_bucket += 1;
            }
        }
    }

    Verdict {
        clip: gold.clip.clone(),
        dtype,
        words_match,
        first_word_diff,
        max_delta_ms,
        beyond_one_bucket,
        shifted_words: shifted,
        malformed,
        out_of_range,
        audio_seconds: gold.audio_seconds,
        elapsed_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn have_gold() -> bool {
        gold_root().join("f16").join("15s_en.json").is_file()
    }

    #[test]
    fn loads_the_frozen_gold() {
        if !have_gold() {
            return;
        }
        let g = GoldJson::load(Dtype::F16, "15s_en").unwrap();
        assert_eq!(g.clip, "15s_en");
        assert_eq!(g.language, "English");
        assert_eq!(g.word_count, g.words.len());
        assert_eq!(g.fixed_ms.len(), g.words.len() * 2);
        assert_eq!(g.raw_masked_ms.len(), g.words.len() * 2);
        assert_eq!(g.timestamp_token_id, 151705);
        assert_eq!(g.timestamp_segment_time_ms, 80.0);
    }

    /// The generator's own numbers must survive the round trip: repairing the
    /// raw buckets has to reproduce `fixed_ms` exactly.
    ///
    /// This is dtype-independent — `fix_timestamps` consumes the argmax buckets
    /// and nothing else — but all three dtypes are checked because each exercises
    /// a different set of interpolation/snap blocks.
    #[test]
    fn fix_timestamps_reproduces_every_gold_fixed_ms() {
        if !have_gold() {
            return;
        }
        for dtype in [Dtype::F16, Dtype::Bf16, Dtype::Fp32] {
            for clip in CLIPS {
                let g = GoldJson::load(dtype, clip).unwrap();
                let got = crate::postprocess::fix_timestamps(&g.raw_masked_ms);
                assert_eq!(got, g.fixed_ms, "{}/{clip}: fix_timestamps diverged", dtype.dir_name());
            }
        }
    }

    #[test]
    fn gold_tsv_regenerates_from_items() {
        if !have_gold() {
            return;
        }
        let g = GoldJson::load(Dtype::F16, "15s_en").unwrap();
        let on_disk = std::fs::read_to_string(gold_root().join("f16").join("15s_en.tsv")).unwrap();
        assert_eq!(g.tsv(), on_disk);
    }

    fn inputs_available() -> bool {
        have_gold() && fixtures_dir().is_dir() && model_dir().join("tokenizer.json").is_file()
    }

    /// The phase-1 gate, end to end and with nothing supplied by hand: the wav
    /// goes in, and the word list and the whole token sequence must come out
    /// exactly as the reference produced them.
    ///
    /// This covers the tokeniser, the CJK/nagisa split, the audio-token length
    /// formula, the mel padding rounded to `2 * n_window`, and the sequence
    /// layout — every part of the input path except the mel/STFT front end,
    /// which `mel.rs` already gates against its own Python dumps.
    #[test]
    fn words_and_input_ids_match_the_reference_for_every_clip() {
        if !inputs_available() {
            return;
        }
        let builder = crate::align_input::InputBuilder::load(&model_dir(), 151705)
            .expect("load tokenizer");
        for clip in CLIPS {
            let gold = GoldJson::load(Dtype::F16, clip).unwrap();
            let wav = fixtures_dir().join(format!("{clip}.wav"));
            let samples = crate::mel::load_audio_wav(&wav, 16000).unwrap().len();
            let valid_mel = crate::align_input::valid_mel_frames(samples);

            let words = crate::words::split_words(&gold.transcript, Some(&gold.language)).unwrap();
            assert_eq!(words, gold.words, "{clip}: word list diverged");

            let input = builder.build(&words, valid_mel).unwrap();
            assert_eq!(
                input.n_audio_tokens, gold.n_audio_tokens,
                "{clip}: audio token count"
            );
            assert_eq!(
                input.padded_mel_frames, gold.mel_frames,
                "{clip}: padded mel width"
            );
            assert_eq!(input.seq_len(), gold.seq_len, "{clip}: sequence length");

            if input.input_ids != gold.input_ids {
                let n = input.input_ids.len().min(gold.input_ids.len());
                let at = (0..n).find(|&i| input.input_ids[i] != gold.input_ids[i]);
                match at {
                    Some(i) => panic!(
                        "{clip}: input_ids differ at {i}: gold={} got={} \
                         (context gold {:?} got {:?})",
                        gold.input_ids[i],
                        input.input_ids[i],
                        &gold.input_ids[i.saturating_sub(3)..(i + 4).min(n)],
                        &input.input_ids[i.saturating_sub(3)..(i + 4).min(n)],
                    ),
                    None => panic!(
                        "{clip}: input_ids lengths differ: gold={} got={}",
                        gold.input_ids.len(),
                        input.input_ids.len()
                    ),
                }
            }
        }
    }
}
