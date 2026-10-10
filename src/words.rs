//! Word splitting — the contract that decides timestamp granularity.
//!
//! Per language: Japanese goes through nagisa; everything else (including
//! Korean) uses content-aware units that match CTC-style alignment:
//!
//! * **Han / CJK ideographs** → one token per character.
//! * **Hangul and Latin** → word-level via spaces, with **script-run** breaks
//!   when Hangul abuts a non-Hangul letter run (and vice versa) with no space.
//! * **Japanese** is morphological via nagisa (`女子` / `アナ` / `の` / `仕事`),
//!   so `ja` needs the `ja` feature.
//!
//! Three behaviours are load-bearing and none are guessable:
//!
//! * **Punctuation is dropped, apostrophes are kept**, and a dropped character
//!   does **not** break the word: `50-minute` becomes the single token
//!   `50minute`.  Only whitespace, a CJK character, or a Hangul↔other letter
//!   script-run boundary ends the current word.
//! * **`_clean_tokens` runs on nagisa's output too**, so a morpheme that is pure
//!   punctuation disappears rather than becoming an empty token.
//! * **Korean is not a pure whitespace split.**  Spaced Hangul eojeol stay
//!   whole, but Hanja / Chinese runs never glue into one blob — each ideograph
//!   is its own token, even under `lang=ko`.

use unicode_general_category::{get_general_category, GeneralCategory};

/// CJK ideographs.  Note `is_kept_char` accepts these a second time: they are
/// `Lo`, so they would pass on the `L` test alone, but the explicit ranges
/// matter because they are also what *ends* a Latin/Hangul run in the default
/// branch (char-level emission).
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

/// Hangul jamo + syllables.  Not ideographs: these stay word-level (space /
/// script-run delimited), unlike Han/CJK which emit per character.
pub fn is_hangul_char(c: char) -> bool {
    let k = c as u32;
    (0x1100..=0x11FF).contains(&k) // Hangul Jamo
        || (0x3130..=0x318F).contains(&k) // Hangul Compatibility Jamo
        || (0xA960..=0xA97F).contains(&k) // Hangul Jamo Extended-A
        || (0xAC00..=0xD7A3).contains(&k) // Hangul Syllables
        || (0xD7B0..=0xD7FF).contains(&k) // Hangul Jamo Extended-B
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

/// Korean: content-aware units (same as [`split_default`]).
///
/// Historically this matched soynlp `LTokenizer()` with empty scores — a pure
/// whitespace split that kept Hanja inside the eojeol.  That glued long Chinese
/// runs into one token whenever ASR mixed ko+zh without spaces.  CTC-style
/// alignment wants Han/CJK at character granularity and Hangul eojeol only when
/// they are space-delimited (or form a Hangul script run).
fn split_korean(text: &str) -> Vec<String> {
    split_default(text)
}

#[cfg(feature = "ja")]
fn japanese_tagger() -> Option<&'static nagisa_rs::Tagger> {
    use std::sync::OnceLock;
    static TAGGER: OnceLock<nagisa_rs::Tagger> = OnceLock::new();
    if let Some(t) = TAGGER.get() {
        return Some(t);
    }
    let t = nagisa_rs::Tagger::embedded().ok()?;
    let _ = TAGGER.set(t);
    TAGGER.get()
}

/// Build the Japanese tagger now, on this thread, if this build has one.
///
/// `Tagger::embedded()` is a **~451 ms** embedded-model load (measured; see
/// `probe_japanese_tagger_cost`), and `split_japanese` would otherwise pay it
/// inside the first Japanese clip — where it is the entire word-splitting cost,
/// since tagging the 420-character `90s_ja` transcript takes only ~20 ms.
///
/// [`crate::align_inference::Aligner::load`] calls this on the thread that loads
/// the BPE, which is already off the load's critical path (the weight upload is
/// the long pole), so the model load happens once *before* any clip instead of
/// once *inside* one.
#[cfg(feature = "ja")]
pub fn warm_japanese_tagger() {
    let _ = japanese_tagger();
}

/// No-op without the `ja` feature: `split_japanese` fails rather than splits.
#[cfg(not(feature = "ja"))]
pub fn warm_japanese_tagger() {}

#[cfg(feature = "ja")]
fn split_japanese(text: &str) -> Result<Vec<String>, WordSplitError> {
    let tagger = japanese_tagger().ok_or(WordSplitError::JapaneseFeatureDisabled)?;
    let words = tagger.tagging(text).words;
    Ok(clean_tokens(words.iter().map(|s| s.as_str())))
}

#[cfg(not(feature = "ja"))]
fn split_japanese(_text: &str) -> Result<Vec<String>, WordSplitError> {
    Err(WordSplitError::JapaneseFeatureDisabled)
}

/// Whether `c` is a letter-like kept char that participates in script-run detection.
/// Numbers and apostrophes attach to the current run without changing its script.
fn is_script_letter(c: char) -> bool {
    c != '\''
        && !matches!(
            get_general_category(c),
            GeneralCategory::DecimalNumber
                | GeneralCategory::LetterNumber
                | GeneralCategory::OtherNumber
        )
}

/// CJK characters individually; Hangul/Latin as space- or script-run-delimited
/// words.  Hangul↔non-Hangul letter transitions flush even without whitespace so
/// glued mixed ASR (`hello안녕`, `사람hello`) does not form one blob.
fn split_default(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    // `None` while the buffer is empty or holds only digits/apostrophes;
    // `Some(true)` once a Hangul letter has set the run; `Some(false)` for
    // any other letter script (Latin, Cyrillic, …).
    let mut buf_hangul: Option<bool> = None;

    for c in text.chars() {
        if is_cjk_char(c) {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
                buf_hangul = None;
            }
            tokens.push(c.to_string());
        } else if c.is_whitespace() {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
                buf_hangul = None;
            }
        } else if is_kept_char(c) {
            if is_script_letter(c) {
                let hangul = is_hangul_char(c);
                if let Some(prev) = buf_hangul {
                    if prev != hangul {
                        tokens.push(std::mem::take(&mut buf));
                    }
                }
                buf_hangul = Some(hangul);
            }
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
        assert_eq!(
            split_words("a 50-minute frame", Some("English")).unwrap(),
            vec!["a", "50minute", "frame"]
        );
    }

    #[test]
    fn apostrophe_is_kept() {
        assert_eq!(
            split_words("don't stop", Some("English")).unwrap(),
            vec!["don't", "stop"]
        );
    }

    #[test]
    fn punctuation_is_dropped_and_does_not_flush() {
        assert_eq!(
            split_words("All right, viewers. So", Some("English")).unwrap(),
            vec!["All", "right", "viewers", "So"]
        );
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
        assert_eq!(
            split_words("用 AI 做", Some("Chinese")).unwrap(),
            vec!["用", "AI", "做"]
        );
    }

    #[test]
    fn no_language_takes_the_default_branch() {
        assert_eq!(split_words("分 钟", None).unwrap(), vec!["分", "钟"]);
    }

    #[test]
    fn combining_marks_are_not_kept() {
        // U+3099 is `Mn` and `Other_Alphabetic`; Python's `category[0] == "L"`
        // is false, so it must not be treated as a letter here either.
        assert_eq!(
            split_words("か\u{3099}き", Some("English")).unwrap(),
            vec!["かき"]
        );
    }

    #[cfg(feature = "ja")]
    #[test]
    fn japanese_uses_morphemes_not_characters() {
        assert_eq!(
            split_words("女子アナの仕事に耐える。", Some("Japanese")).unwrap(),
            vec!["女子", "アナ", "の", "仕事", "に", "耐える"]
        );
    }

    #[test]
    fn korean_is_whitespace_split_for_eojeol() {
        // Spaced Hangul eojeol stay whole (soynlp-empty-scores shape), but the
        // path is now content-aware — see mixed / Hanja tests below.
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
    fn korean_hanja_is_char_level_not_glued() {
        // Han/CJK must never stay as one blob under lang=ko.
        assert_eq!(
            split_words("中國 사람", Some("Korean")).unwrap(),
            vec!["中", "國", "사람"]
        );
        assert_eq!(
            split_words("中國사람", Some("Korean")).unwrap(),
            vec!["中", "國", "사람"]
        );
    }

    #[test]
    fn glued_korean_chinese_splits_by_script_runs() {
        assert_eq!(
            split_words("안녕하세요你好世界", Some("Korean")).unwrap(),
            vec!["안녕하세요", "你", "好", "世", "界"]
        );
        // Same content under Chinese / default — mixed sharpness is not ko-only.
        assert_eq!(
            split_words("안녕하세요你好世界", Some("Chinese")).unwrap(),
            vec!["안녕하세요", "你", "好", "世", "界"]
        );
        assert_eq!(
            split_words("안녕하세요你好世界", None).unwrap(),
            vec!["안녕하세요", "你", "好", "世", "界"]
        );
    }

    #[test]
    fn mixed_ko_zh_en_script_runs() {
        assert_eq!(
            split_words("사람你好hello世界", Some("Korean")).unwrap(),
            vec!["사람", "你", "好", "hello", "世", "界"]
        );
        assert_eq!(
            split_words("hello안녕", Some("English")).unwrap(),
            vec!["hello", "안녕"]
        );
        assert_eq!(
            split_words("안녕hello", Some("Korean")).unwrap(),
            vec!["안녕", "hello"]
        );
        // ISO code aliases
        assert_eq!(
            split_words("hello世界ko테스트", Some("ko")).unwrap(),
            vec!["hello", "世", "界", "ko", "테스트"]
        );
    }

    #[test]
    fn real_asr_raw_mixed_ko_zh_en_snippet() {
        // Realistic ASR raw: Korean eojeol, then a glued Chinese run with no
        // spaces, Hangul name abutting the last Han char, then Latin.
        let asr_raw = "네 맞아요这个就是我们要找的人김민수 씨입니다 hello everyone";
        let got = split_words(asr_raw, Some("Korean")).unwrap();
        assert_eq!(
            got,
            vec![
                "네",
                "맞아요",
                "这",
                "个",
                "就",
                "是",
                "我",
                "们",
                "要",
                "找",
                "的",
                "人",
                "김민수",
                "씨입니다",
                "hello",
                "everyone",
            ]
        );
        // No token is a multi-character Han run.
        for t in &got {
            let han: String = t.chars().filter(|&c| is_cjk_char(c)).collect();
            assert!(
                han.chars().count() <= 1,
                "token {t:?} glued Han run {han:?}"
            );
        }
    }

    #[test]
    fn ko_lang_does_not_disable_mixed_smart_split() {
        // lang=ko must not fall back to pure whitespace (old split_korean).
        // 중/국 are Hangul syllables, so unspaced "오늘중국" is one Hangul run;
        // only 北/京 are char-level ideographs.
        let glued = "오늘중국北京여행OK";
        assert_eq!(
            split_words(glued, Some("Korean")).unwrap(),
            vec!["오늘중국", "北", "京", "여행", "OK"]
        );
        assert_eq!(
            split_words("오늘中國여행OK", Some("ko")).unwrap(),
            vec!["오늘", "中", "國", "여행", "OK"]
        );
    }

    /// Probe, not a gate: where the Japanese path's time actually goes.
    ///
    /// `split_japanese` builds the tagger lazily, on the first call, and the
    /// aligner calls it once per clip — so the tagger's construction is charged
    /// to `words_ms` on the one clip that uses it.  If the numbers say the
    /// construction is the cost, the fix is to warm it at load, off the clip;
    /// if it is the tagging, the fix has to be somewhere else entirely.
    ///
    /// Run with `--nocapture`; it needs the fixture and the frozen gold (for the
    /// transcript) and skips silently without them.
    #[cfg(feature = "ja")]
    #[test]
    fn probe_japanese_tagger_cost() {
        use std::time::Instant;
        let gold = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tools/gold/fp32/90s_ja.json");
        if !gold.is_file() {
            return;
        }
        let Ok(text) = std::fs::read_to_string(&gold).map(|s| {
            let v: serde_json::Value = serde_json::from_str(&s).unwrap();
            v["transcript"].as_str().unwrap().to_string()
        }) else {
            return;
        };

        let t = Instant::now();
        let tagger = nagisa_rs::Tagger::embedded().expect("embedded tagger");
        let init = t.elapsed();
        let t = Instant::now();
        let first = tagger.tagging(&text).words.len();
        let first_t = t.elapsed();
        let t = Instant::now();
        let second = tagger.tagging(&text).words.len();
        let second_t = t.elapsed();
        eprintln!(
            "nagisa probe: {} chars -> {first}/{second} morphemes | \
             Tagger::embedded() {:.1} ms | first tagging {:.1} ms | second tagging {:.1} ms",
            text.chars().count(),
            init.as_secs_f64() * 1000.0,
            first_t.as_secs_f64() * 1000.0,
            second_t.as_secs_f64() * 1000.0,
        );
    }
}
