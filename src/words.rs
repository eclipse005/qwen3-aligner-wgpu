//! Word splitting — the contract that decides timestamp granularity.
//!
//! Port of `Qwen3ASRProcessor.split_words_for_alignment`
//! (`transformers/models/qwen3_asr/processing_qwen3_asr.py`), which is itself a
//! port of `Qwen3ForceAlignProcessor.encode_timestamp` in
//! `D:\Qwen3-ASR\qwen_asr\inference\qwen3_forced_aligner.py`.
//!
//! The two agree on every fixture (the gold generator refuses to freeze a clip
//! whose word list the two disagree on), so either is a safe spec; this follows
//! the transformers one because that is the API the gold was produced through.
//!
//! Three behaviours here are load-bearing and none are guessable:
//!
//! * **Japanese is not per-character.**  nagisa's morphemes are the units
//!   (`女子` / `アナ` / `の` / `仕事`), so `ja` needs a morphological analyser and
//!   the `ja` feature.  Chinese, by contrast, *is* one token per CJK character.
//! * **Punctuation is dropped, apostrophes are kept**, and a dropped character
//!   does **not** break the word: `50-minute` becomes the single token
//!   `50minute`.  Only whitespace and a CJK character end the current word.
//! * **`_clean_tokens` runs on nagisa's output too**, so a morpheme that is pure
//!   punctuation disappears rather than becoming an empty token.

use unicode_general_category::{get_general_category, GeneralCategory};

/// CJK ideographs.  Note `is_kept_char` accepts these a second time: they are
/// `Lo`, so they would pass on the `L` test alone, but the explicit ranges
/// matter because they are also what *ends* a Latin run in the default branch.
pub fn is_cjk_char(c: char) -> bool {
    let k = c as u32;
    (0x4E00..=0x9FFF).contains(&k)
        || (0x3400..=0x4DBF).contains(&k)
        || (0x20000..=0x2A6DF).contains(&k)
        || (0x2A700..=0x2B73F).contains(&k)
        || (0x2B740..=0x2B81F).contains(&k)
        || (0x2B820..=0x2CEAF).contains(&k)
        || (0xF900..=0xFAFF).contains(&k)
        || (0x2F800..=0x2FA1F).contains(&k)
}

/// Characters that survive tokenisation: letters, numbers, apostrophes, CJK.
///
/// This is `unicodedata.category(ch)[0] in "LN"` — the *general category*, not
/// the derived `Alphabetic` property.  `char::is_alphabetic()` would be wrong:
/// it also accepts `Other_Alphabetic` combining marks (`Mn`/`Mc`, e.g. the
/// voiced sound mark U+3099), which Python drops.
pub fn is_kept_char(c: char) -> bool {
    if c == '\'' {
        return true;
    }
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
            | GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    ) || is_cjk_char(c)
}

/// Keep only [`is_kept_char`] characters, dropping the token if nothing survives.
fn clean_token(token: &str) -> Option<String> {
    let cleaned: String = token.chars().filter(|&c| is_kept_char(c)).collect();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn clean_tokens<'a, I: IntoIterator<Item = &'a str>>(raw: I) -> Vec<String> {
    raw.into_iter().filter_map(clean_token).collect()
}

#[derive(Debug, thiserror::Error)]
pub enum WordSplitError {
    #[error(
        "language {0:?} is not supported by the forced aligner; \
         supported: Chinese, Cantonese, English, French, German, Italian, \
         Japanese, Korean, Portuguese, Russian, Spanish"
    )]
    UnsupportedLanguage(String),
    #[error("Japanese word splitting requires building with the `ja` feature (nagisa_rs)")]
    JapaneseFeatureDisabled,
}

/// Split a transcript into the exact units the model emits two timestamps for.
///
/// `language` accepts full names or ISO codes, case-insensitively, matching
/// `prepare_language_inputs`; anything unrecognised takes the default branch
/// rather than erroring, exactly like the reference.
pub fn split_words(text: &str, language: Option<&str>) -> Result<Vec<String>, WordSplitError> {
    let text = text.trim();
    let lang = language.unwrap_or("").to_lowercase();

    if lang == "japanese" || lang == "ja" {
        return split_japanese(text);
    }
    if lang == "korean" || lang == "ko" {
        return Ok(split_korean(text));
    }
    Ok(split_default(text))
}

/// Korean: `_clean_tokens(LTokenizer().tokenize(text))`, with the **default empty
/// scores**.
///
/// Measured, not assumed: with no score dictionary, soynlp's `LTokenizer` splits
/// on nothing — every whitespace-delimited run comes back whole
/// (`안녕하세요 오늘 날씨가 좋네요` -> those same four words).  Giving it the
/// 17 968-entry `korean_dict_jieba.dict` instead produces `안녕 / 하세요`,
/// `날씨 / 가`, which is a *different* word list and therefore a different
/// timestamp granularity.
///
/// Note this is deliberately **not** `split_default`: Korean text can carry
/// hanja, and `clean_token` keeps a CJK character inside its word rather than
/// emitting it as its own token the way the Chinese path does.
fn split_korean(text: &str) -> Vec<String> {
    clean_tokens(text.split_whitespace())
}

#[cfg(feature = "ja")]
fn split_japanese(text: &str) -> Result<Vec<String>, WordSplitError> {
    use std::sync::OnceLock;
    static TAGGER: OnceLock<nagisa_rs::Tagger> = OnceLock::new();
    let tagger = match TAGGER.get() {
        Some(t) => t,
        None => {
            let t = nagisa_rs::Tagger::embedded()
                .map_err(|_| WordSplitError::JapaneseFeatureDisabled)?;
            let _ = TAGGER.set(t);
            TAGGER.get().expect("tagger just set")
        }
    };
    let words = tagger.tagging(text).words;
    Ok(clean_tokens(words.iter().map(|s| s.as_str())))
}

#[cfg(not(feature = "ja"))]
fn split_japanese(_text: &str) -> Result<Vec<String>, WordSplitError> {
    Err(WordSplitError::JapaneseFeatureDisabled)
}

/// CJK characters individually; space-delimited scripts produce whole words.
fn split_default(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for c in text.chars() {
        if is_cjk_char(c) {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(c.to_string());
        } else if c.is_whitespace() {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
        } else if is_kept_char(c) {
            buf.push(c);
        }
        // Anything else (punctuation, symbols, marks) is dropped *without*
        // flushing: that is why `50-minute` is one token.
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyphen_does_not_split_a_word() {
        assert_eq!(split_words("a 50-minute frame", Some("English")).unwrap(),
                   vec!["a", "50minute", "frame"]);
    }

    #[test]
    fn apostrophe_is_kept() {
        assert_eq!(split_words("don't stop", Some("English")).unwrap(),
                   vec!["don't", "stop"]);
    }

    #[test]
    fn punctuation_is_dropped_and_does_not_flush() {
        assert_eq!(split_words("All right, viewers. So", Some("English")).unwrap(),
                   vec!["All", "right", "viewers", "So"]);
    }

    #[test]
    fn chinese_is_one_token_per_character() {
        // Note the transcript really does contain 在 twice: 现在 + 在天海酒吧.
        assert_eq!(
            split_words("我现在在天海酒吧，限你十分钟。", Some("Chinese")).unwrap(),
            vec!["我", "现", "在", "在", "天", "海", "酒", "吧", "限", "你", "十", "分", "钟"]
        );
    }

    #[test]
    fn latin_run_inside_chinese_is_one_token() {
        assert_eq!(split_words("用 AI 做", Some("Chinese")).unwrap(),
                   vec!["用", "AI", "做"]);
    }

    #[test]
    fn no_language_takes_the_default_branch() {
        assert_eq!(split_words("分 钟", None).unwrap(), vec!["分", "钟"]);
    }

    #[test]
    fn combining_marks_are_not_kept() {
        // U+3099 is `Mn` and `Other_Alphabetic`; Python's `category[0] == "L"`
        // is false, so it must not be treated as a letter here either.
        assert_eq!(split_words("か\u{3099}き", Some("English")).unwrap(), vec!["かき"]);
    }

    #[cfg(feature = "ja")]
    #[test]
    fn japanese_uses_morphemes_not_characters() {
        assert_eq!(split_words("女子アナの仕事に耐える。", Some("Japanese")).unwrap(),
                   vec!["女子", "アナ", "の", "仕事", "に", "耐える"]);
    }

    #[test]
    fn korean_is_whitespace_split() {
        // Verified against soynlp's `LTokenizer()` with its default empty scores.
        assert_eq!(
            split_words("안녕하세요 오늘 날씨가 좋네요", Some("Korean")).unwrap(),
            vec!["안녕하세요", "오늘", "날씨가", "좋네요"]
        );
        assert_eq!(
            split_words("그 사람은 정말 친절했습니다.", Some("Korean")).unwrap(),
            vec!["그", "사람은", "정말", "친절했습니다"]
        );
    }

    #[test]
    fn korean_keeps_hanja_inside_its_word() {
        // The Chinese path would emit 中 and 國 separately; the Korean path must not.
        assert_eq!(split_words("中國 사람", Some("Korean")).unwrap(), vec!["中國", "사람"]);
    }
}
