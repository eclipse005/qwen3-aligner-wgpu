//! Input construction — words and audio become one token sequence.
//!
//! The exact shape, dumped from the reference on `15s_en`:
//!
//! ```text
//! [151669]  <|audio_start|>                     x1
//! [151676]  <|audio_pad|>                       xN   N = audio_token_count(mel)
//! [151670]  <|audio_end|>                       x1
//! w0 [151705][151705] w1 [151705][151705] ...  2 timestamps per word
//! ```
//!
//! **No BOS, no EOS, no chat template.**  The upstream wrapper builds this string
//! by hand and hands it to the processor, which only expands the single
//! `<|audio_pad|>` into `N` copies ("replace_audio_token").  For `15s_en`:
//! `323 = 1 + 195 + 1 + 78 + 47`, i.e. the 39 words cost 47 BPE tokens — one
//! word is *not* one token, and only the `<timestamp>` count is tied to the word
//! count.
//!
//! Two counts come out of the audio side and they are **not** the same number:
//!
//! * `valid_mel_frames` is what the feature extractor's attention mask sums to,
//!   `floor(n_samples / hop)`, and it is what drives `audio_token_count`;
//! * `padded_mel_frames` is that rounded up to a multiple of `n_window * 2`, the
//!   width of the mel array the encoder consumes.
//!
//! The padding is `np.pad`'s default constant **0.0**, not the log-mel floor
//! (`(max-4)/4`) — the padded region is a different value and the encoder sees it.

use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// `<|audio_start|>` in the shipped checkpoint.
pub const AUDIO_START_TOKEN: &str = "<|audio_start|>";
/// `<|audio_end|>`.
pub const AUDIO_END_TOKEN: &str = "<|audio_end|>";
/// `<|audio_pad|>` — the placeholder the processor expands.
pub const AUDIO_TOKEN: &str = "<|audio_pad|>";
/// The per-word timestamp marker.  Not registered as a special token, but it is
/// the last entry of the vocabulary and the BPE resolves it atomically.
pub const TIMESTAMP_TOKEN: &str = "<timestamp>";

/// Mel hop in samples — the STFT's, not a second copy of it.
pub(crate) use crate::mel::HOP_LENGTH;
/// Output frames per full chunk: the conv stem reduces 100 mel frames to 13.
pub const TOKENS_PER_CHUNK: usize = 13;

/// `(L - 1) / 2 + 1` applied three times, the length after three k=3/s=2/p=1
/// convolutions.  Zero stays zero (Python would floor-divide a negative).
pub fn conv3_length(len: usize) -> usize {
    let mut l = len as isize;
    for _ in 0..3 {
        l = if l > 0 { (l - 1) / 2 + 1 } else { 0 };
    }
    l as usize
}

/// How many `<|audio_pad|>` tokens the processor will emit for this clip.
///
/// `Qwen3ASRProcessor._get_audio_token_length`: full 100-frame chunks contribute
/// 13 tokens each, and the trailing partial chunk goes through the conv stack.
pub fn audio_token_count(valid_mel_frames: usize, n_window: usize) -> usize {
    let chunk_len = n_window * 2;
    let remainder = valid_mel_frames % chunk_len;
    (valid_mel_frames / chunk_len) * TOKENS_PER_CHUNK + conv3_length(remainder)
}

/// The mel array width: `valid_mel_frames` right-padded to a multiple of
/// `n_window * 2`.
pub fn padded_mel_frames(valid_mel_frames: usize, n_window: usize) -> usize {
    let multiple = n_window * 2;
    let remainder = valid_mel_frames % multiple;
    if remainder == 0 {
        valid_mel_frames
    } else {
        valid_mel_frames + (multiple - remainder)
    }
}

/// Valid mel frames from a sample count: `floor(n_samples / hop)`.
///
/// `torch.stft(center=True)` yields `1 + floor(n_samples / hop)` frames and the
/// feature extractor drops the last (`stft[..., :-1]`).
pub fn valid_mel_frames(n_samples: usize) -> usize {
    n_samples / HOP_LENGTH
}

/// The assembled model input, plus the facts a caller needs to sanity-check it.
#[derive(Debug, Clone)]
pub struct AlignerInput {
    pub input_ids: Vec<u32>,
    pub words: Vec<String>,
    pub n_audio_tokens: usize,
    pub valid_mel_frames: usize,
    pub padded_mel_frames: usize,
    pub audio_token_id: u32,
    /// Indices into `input_ids` where the model's output is read: the
    /// `<timestamp>` positions, in order, `(start_0, end_0, start_1, ...)`.
    pub timestamp_positions: Vec<usize>,
}

impl AlignerInput {
    pub fn seq_len(&self) -> usize {
        self.input_ids.len()
    }

    /// Positions of the audio frames inside `input_ids`, for the scatter step.
    pub fn audio_positions(&self) -> Vec<usize> {
        self.input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.audio_token_id)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Builds the input sequence.  Owns the BPE tokenizer from the checkpoint.
pub struct InputBuilder {
    tokenizer: Tokenizer,
    n_window: usize,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
    pub timestamp_token_id: u32,
    pub vocab_size: usize,
}

impl InputBuilder {
    /// Load `tokenizer.json` plus the marker strings from the checkpoint's
    /// `tokenizer_config.json`.
    pub fn load(model_dir: &Path, timestamp_token_id: u32) -> Result<Self> {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", tok_path.display()))?;
        let vocab_size = tokenizer.get_vocab_size(true);

        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
                .context("tokenizer_config.json")?,
        )?;
        let marker = |key: &str, default: &str| -> String {
            cfg.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or(default)
                .to_string()
        };
        let audio_token = marker("audio_token", AUDIO_TOKEN);
        let audio_bos = marker("audio_bos_token", AUDIO_START_TOKEN);
        let audio_eos = marker("audio_eos_token", AUDIO_END_TOKEN);

        let id_of = |s: &str| -> Result<u32> {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| anyhow::anyhow!("tokenizer has no {s:?}"))
        };

        // The processor's window comes from the feature extractor; 50 for every
        // shipped checkpoint, asserted against config.json by the caller.
        let n_window = 50;

        Ok(Self {
            audio_start_token_id: id_of(&audio_bos)?,
            audio_end_token_id: id_of(&audio_eos)?,
            audio_token_id: id_of(&audio_token)?,
            timestamp_token_id,
            vocab_size,
            n_window,
            tokenizer,
        })
    }

    pub fn n_window(&self) -> usize {
        self.n_window
    }

    /// Encode a raw string the way the reference does: one pass, no special
    /// tokens added (the checkpoint's post-processor adds none, and the dumped
    /// sequence starts straight at `<|audio_start|>`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Assemble the model input for one sample.
    pub fn build(&self, words: &[String], valid_mel: usize) -> Result<AlignerInput> {
        let n_audio_tokens = audio_token_count(valid_mel, self.n_window);
        anyhow::ensure!(
            n_audio_tokens > 0,
            "audio produced 0 tokens ({valid_mel} mel frames)"
        );

        let mut text = String::with_capacity(n_audio_tokens * 12 + words.len() * 22 + 32);
        text.push_str(AUDIO_START_TOKEN);
        for _ in 0..n_audio_tokens {
            text.push_str(AUDIO_TOKEN);
        }
        text.push_str(AUDIO_END_TOKEN);
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                text.push_str(TIMESTAMP_TOKEN);
                text.push_str(TIMESTAMP_TOKEN);
            }
            text.push_str(w);
        }
        text.push_str(TIMESTAMP_TOKEN);
        text.push_str(TIMESTAMP_TOKEN);

        let input_ids = self.encode(&text)?;

        // Structural self-check: the encoder's scatter and the timestamp gather
        // both index by these ids, so a mismatch here is a silent wrong answer
        // two phases later.
        let audio_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.audio_token_id)
            .map(|(i, _)| i)
            .collect();
        anyhow::ensure!(
            audio_positions.len() == n_audio_tokens,
            "expected {n_audio_tokens} audio tokens, tokenizer produced {}",
            audio_positions.len()
        );
        anyhow::ensure!(
            input_ids[0] == self.audio_start_token_id,
            "sequence must start with <|audio_start|>, got {}",
            input_ids[0]
        );
        anyhow::ensure!(
            input_ids.get(n_audio_tokens + 1) == Some(&self.audio_end_token_id),
            "expected <|audio_end|> right after the audio tokens"
        );

        let timestamp_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.timestamp_token_id)
            .map(|(i, _)| i)
            .collect();
        anyhow::ensure!(
            timestamp_positions.len() == words.len() * 2,
            "expected {} timestamp tokens for {} words, got {}",
            words.len() * 2,
            words.len(),
            timestamp_positions.len()
        );

        Ok(AlignerInput {
            input_ids,
            words: words.to_vec(),
            n_audio_tokens,
            valid_mel_frames: valid_mel,
            padded_mel_frames: padded_mel_frames(valid_mel, self.n_window),
            audio_token_id: self.audio_token_id,
            timestamp_positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv3_matches_the_python_length_formula() {
        // (L-1)//2+1 three times; Python floors, so 0 must not wrap.
        assert_eq!(conv3_length(0), 0);
        assert_eq!(conv3_length(1), 1);
        assert_eq!(conv3_length(2), 1);
        assert_eq!(conv3_length(9), 2);
        assert_eq!(conv3_length(32), 4);
        assert_eq!(conv3_length(100), 13);
    }

    /// Every fixture's `n_audio_tokens`, recomputed from its sample count, which
    /// is itself recovered from the source wav so nothing is taken on trust.
    #[test]
    fn audio_token_count_matches_every_gold() {
        let fixtures = crate::gold::fixtures_dir();
        if !fixtures.is_dir() {
            return;
        }
        for (clip, want) in [
            ("15s_en", 195usize),
            ("30s_zh", 392),
            ("90s_ja", 1161),
            ("90s_en", 1170),
            ("180s_en", 2292),
            ("180s_zh", 2340),
        ] {
            let samples = crate::mel::load_audio_wav(fixtures.join(format!("{clip}.wav")), 16000)
                .unwrap()
                .len();
            let valid = valid_mel_frames(samples);
            assert_eq!(
                audio_token_count(valid, 50),
                want,
                "{clip}: {samples} samples -> {valid} valid mel frames"
            );
        }
    }

    #[test]
    fn padded_mel_is_a_multiple_of_two_windows() {
        assert_eq!(padded_mel_frames(1500, 50), 1500);
        assert_eq!(padded_mel_frames(3009, 50), 3100);
        assert_eq!(padded_mel_frames(8932, 50), 9000);
        assert_eq!(padded_mel_frames(9000, 50), 9000);
        assert_eq!(padded_mel_frames(17630, 50), 17700);
        assert_eq!(padded_mel_frames(18000, 50), 18000);
    }
}
