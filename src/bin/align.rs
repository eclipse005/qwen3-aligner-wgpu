//! `align` — forced alignment CLI.
//!
//! Current stage: the **input path only**.  `--check-input` runs wav -> words ->
//! `input_ids` and diffs the result against the frozen gold; the model forward is
//! not wired in yet, so `--emit` refuses rather than printing timestamps it
//! cannot substantiate.
//!
//! ```text
//! align --check-input --all
//! align --check-input --wav 90s_ja
//! align --list-clips
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use qwen3_aligner_wgpu::align_input::{self, InputBuilder};
use qwen3_aligner_wgpu::gold::{self, Dtype, GoldJson};
use qwen3_aligner_wgpu::words;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-devices") {
        for d in pollster::block_on(qwen3_aligner_wgpu::list_devices()) {
            println!("{}", d.describe());
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-clips") {
        for (clip, lang) in gold::CLIPS.iter().zip(gold::LANGUAGES) {
            println!("{clip}\t{lang}");
        }
        return Ok(());
    }

    let model_dir = arg_value(&args, "--model")
        .map(PathBuf::from)
        .unwrap_or_else(gold::model_dir);
    let dtype = arg_value(&args, "--dtype")
        .map(|s| Dtype::parse(&s).unwrap_or_else(|| panic!("unknown --dtype {s}")))
        .unwrap_or(Dtype::F16);

    if !args.iter().any(|a| a == "--check-input") {
        if args.iter().any(|a| a == "--eval") {
            return eval(args, model_dir);
        }
        return run(args, model_dir, dtype);
    }

    let clips: Vec<&str> = if args.iter().any(|a| a == "--all") {
        gold::CLIPS.to_vec()
    } else {
        vec![arg_value(&args, "--wav")
            .context("--wav <clip> or --all is required")?
            .leak() as &str]
    };

    let cfg = qwen3_aligner_wgpu::config::AsrConfig::from_file(&model_dir.join("config.json"))
        .with_context(|| format!("load config from {}", model_dir.display()))?;
    let head = cfg
        .align
        .as_ref()
        .context("checkpoint has no forced-aligner head (timestamp_token_id missing)")?;
    let n_window = cfg.thinker_config.audio_config.n_window;
    anyhow::ensure!(
        n_window == 50,
        "audio_token_count assumes n_window=50, checkpoint says {n_window}"
    );

    let builder = InputBuilder::load(&model_dir, head.timestamp_token_id as u32)?;
    let fixtures = gold::fixtures_dir();

    println!(
        "model    : {}  (labels={} segment={}ms)",
        model_dir.display(),
        head.num_labels,
        head.timestamp_segment_time_ms
    );
    println!("gold     : {} / {}", gold::gold_root().display(), dtype.dir_name());
    println!();

    let mut failed = 0usize;
    for clip in clips {
        match check_one(&builder, &fixtures, clip, dtype) {
            Ok(line) => println!("{line}"),
            Err(e) => {
                failed += 1;
                println!("{clip:<9} ERROR   {e:#}");
            }
        }
    }
    if failed > 0 {
        bail!("{failed} clip(s) failed");
    }
    println!("\nall clips match the reference input exactly");
    Ok(())
}

fn check_one(builder: &InputBuilder, fixtures: &std::path::Path, clip: &str, dtype: Dtype) -> Result<String> {
    let t0 = Instant::now();
    let gold = GoldJson::load(dtype, clip)?;
    let wav = fixtures.join(format!("{clip}.wav"));
    let samples = qwen3_aligner_wgpu::load_audio_wav(&wav, 16000)?
        .len();
    let valid_mel = align_input::valid_mel_frames(samples);

    let words = words::split_words(&gold.transcript, Some(&gold.language))?;
    let input = builder.build(&words, valid_mel)?;
    let dt = t0.elapsed().as_secs_f64();

    let words_ok = words == gold.words;
    let ids_ok = input.input_ids == gold.input_ids;
    let tok_ok = input.n_audio_tokens == gold.n_audio_tokens;
    let mel_ok = input.padded_mel_frames == gold.mel_frames;

    let status = if words_ok && ids_ok && tok_ok && mel_ok {
        "OK"
    } else {
        "MISMATCH"
    };
    let mut line = format!(
        "{clip:<9} {status:<8} words={:<4} seq={:<5} audio={:<5} mel={:<6} \
         ({:.3}s)",
        words.len(),
        input.seq_len(),
        input.n_audio_tokens,
        input.padded_mel_frames,
        dt,
    );
    if !words_ok {
        let i = words
            .iter()
            .zip(&gold.words)
            .position(|(a, b)| a != b)
            .unwrap_or(words.len().min(gold.words.len()));
        line.push_str(&format!(
            "\n    words differ at {i}: gold={:?} got={:?}",
            gold.words.get(i),
            words.get(i)
        ));
    }
    if !ids_ok {
        let i = input
            .input_ids
            .iter()
            .zip(&gold.input_ids)
            .position(|(a, b)| a != b)
            .unwrap_or(input.input_ids.len().min(gold.input_ids.len()));
        line.push_str(&format!(
            "\n    input_ids differ at {i}: gold={} got={}",
            gold.input_ids.get(i).map(|v| v.to_string()).unwrap_or("<>".into()),
            input
                .input_ids
                .get(i)
                .map(|v| v.to_string())
                .unwrap_or("<>".into())
        ));
    }
    if !tok_ok {
        line.push_str(&format!(
            "\n    audio tokens: gold={} got={}",
            gold.n_audio_tokens, input.n_audio_tokens
        ));
    }
    if !mel_ok {
        line.push_str(&format!(
            "\n    padded mel: gold={} got={}",
            gold.mel_frames, input.padded_mel_frames
        ));
    }
    Ok(line)
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// One clip of the multi-language FLEURS eval (`tools/eval/<config>/<clip>.json`),
/// written by `D:\Qwen3-ASR\align_eval.py`.
#[derive(serde::Deserialize)]
struct EvalClip {
    clip: String,
    wav: String,
    language: String,
    transcript: String,
    words: Vec<String>,
    /// The argmax buckets before `_fix_timestamps` touched them.
    raw_masked_ms: Vec<i64>,
    fixed_ms: Vec<i64>,
    audio_seconds: f64,
    /// The reference's own top-1 minus top-2 logit at each timestamp position.
    /// Near zero means the reference chose that bucket by floating-point luck.
    #[serde(default)]
    margin: Vec<f32>,
}

/// Run the port over the whole FLEURS eval and report per language.
///
/// The gate is the same one the six fixtures use, and it is applied per *clip*:
/// the word list must match verbatim (the tokeniser decides it), and every
/// timestamp must land within one 80 ms bucket of the reference.
fn eval(args: Vec<String>, model_dir: PathBuf) -> Result<()> {
    use qwen3_aligner_wgpu::align_inference::Aligner;
    use qwen3_aligner_wgpu::gold::{TsVerdict, MARGIN_NOISE_FLOOR};
    use qwen3_aligner_wgpu::gpu::DeviceSelector;

    let spec = arg_value(&args, "--device").unwrap_or_else(|| "auto".to_string());
    let root = gold::gold_root().parent().unwrap().join("eval");
    let only = arg_value(&args, "--lang");
    let limit: Option<usize> = arg_value(&args, "--limit").and_then(|s| s.parse().ok());

    let mut aligner = Aligner::load(DeviceSelector::parse(&spec)?, &model_dir)?;
    println!("device : {}", aligner.describe());
    println!("eval   : {}", root.display());
    println!();
    println!("{:<14} {:<11} {:>5} {:>7} {:>9} {:>10} {:>9} {:>8} {:>8}",
             "config", "language", "clips", "words", "raw-exact", "raw-maxdelta",
             "ts-exact", "ts-maxdelta", "RTFx");

    // (clips, languages covered, languages not covered, words,
    //  raw-exact clips, ts-exact clips, clips failing the margin gate)
    let mut covering = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut worst_raw = 0i64;
    let mut worst_ts = 0i64;
    let mut excused_total = 0usize;
    let mut excused_worst = 0.0f32;
    let mut failures: Vec<String> = Vec::new();
    let mut amplified: Vec<String> = Vec::new();
    let mut probes = 0usize;

    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .with_context(|| format!("read {}", root.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    dirs.sort();

    for dir in dirs {
        let config = dir.file_name().unwrap().to_string_lossy().to_string();
        if let Some(l) = &only {
            if &config != l {
                continue;
            }
        }
        let mut files: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .map(|e| e.path())
            .collect();
        files.sort();
        // The ASR round's methodology: a fixed sample per language, not the whole
        // split.  The full FLEURS gold is on disk (7 600+ clips) and can be
        // spot-checked later, but verifying all of it costs ~an hour of GPU time
        // to re-derive a number that 20 clips per language already establishes.
        if let Some(n) = limit {
            files.truncate(n);
        }

        let mut clips = 0usize;
        let mut words = 0usize;
        let mut raw_exact = 0usize;
        let mut ts_exact = 0usize;

        let mut lang_raw_max = 0i64;
        let mut lang_ts_max = 0i64;
        let mut audio_s = 0.0f64;
        let mut elapsed_s = 0.0f64;
        let mut language = String::new();
        let mut unsupported: Option<String> = None;

        for path in files {
            let c: EvalClip = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
            language = c.language.clone();
            clips += 1;
            audio_s += c.audio_seconds;

            let t0 = Instant::now();
            let samples = qwen3_aligner_wgpu::load_audio_wav(&c.wav, 16000)?;
            let (mel, _b, _f) = qwen3_aligner_wgpu::mel::mel_features(&samples)?;
            let valid = align_input::valid_mel_frames(samples.len());
            let padded = align_input::padded_mel_frames(valid, aligner.config().audio_cfg.n_window);
            let mut mel_padded = mel;
            mel_padded.resize(128 * padded, 0.0);
            // A language the tokeniser cannot handle yet is a *gap in coverage*,
            // not a clip failure: report it as such instead of aborting the run
            // and hiding the nine languages that do work.
            let words_run = match words::split_words(&c.transcript, Some(&c.language)) {
                Ok(w) => w,
                Err(e) => {
                    unsupported = Some(e.to_string());
                    break;
                }
            };
            let input = build_input_with(&model_dir, aligner.config().timestamp_token_id, &words_run, valid)?;
            let raw = aligner.align_raw_ms(&mel_padded, valid, &input)?;
            elapsed_s += t0.elapsed().as_secs_f64();

            words += c.words.len();

            // A clip with no frozen reference is a *probe*, not a pass: the
            // question there is whether the port completes at all, and counting
            // an empty comparison as "exact" would manufacture agreement.  It is
            // checked before the word list because a probe carries no gold words
            // to compare against -- the run supplies its own.
            if c.raw_masked_ms.is_empty() {
                probes += 1;
                println!("      probe {:<16} words={:<5} seq={:<6} ts={:<5} last={}ms",
                         c.clip, words_run.len(), input.seq_len(), raw.len(),
                         raw.last().copied().unwrap_or(-1));
                continue;
            }

            if words_run != c.words {

                covering.6 += 1;
                failures.push(format!("{}: word list differs ({} vs {})",
                                      c.clip, words_run.len(), c.words.len()));
                continue;
            }

            // Axis 2: the raw argmax.  This is what the towers actually produced,
            // and it is the only honest measure of numerical agreement.
            let raw_delta = TsVerdict::of(&raw, &c.raw_masked_ms, &c.margin);
            // Axis 3: the reported output.  `_fix_timestamps` is discontinuous, so
            // a single moved bucket can come out the far end as several.
            let fixed = qwen3_aligner_wgpu::fix_timestamps(&raw);
            let ts_delta = TsVerdict::of(&fixed, &c.fixed_ms, &c.margin);

            if raw_delta.moved == 0 {
                raw_exact += 1;
            }
            if ts_delta.moved == 0 {
                ts_exact += 1;
            }
            // The gate is on the **raw** comparison, because that is where the
            // index alignment is exact: `margin[i]` describes the reference's
            // argmax at position `i`.  After `_fix_timestamps` the value at `i`
            // can come from elsewhere entirely, so a moved *reported* value is a
            // consequence of a moved raw value and has to be traced back to it
            // rather than judged against `margin[i]`.
            if raw_delta.beyond_floor > 0 {
                covering.6 += 1;
                failures.push(format!(
                    "{}: {} raw endpoint(s) moved at a confident position (margin > {}), \
                     worst {}ms, smallest margin among them {:.5}",
                    c.clip,
                    raw_delta.beyond_floor,
                    MARGIN_NOISE_FLOOR,
                    raw_delta.max_delta_ms,
                    raw_delta.worst_confident_margin
                ));
            }
            // `fix_timestamps` is deterministic, so identical raw input must give
            // identical output.  If that ever fails, the repair is reading
            // something outside its argument.
            if ts_delta.moved > 0 && raw_delta.moved == 0 {
                failures.push(format!(
                    "{}: raw matches everywhere but {} reported endpoint(s) moved — \
                     the repair is not a function of the raw values alone",
                    c.clip, ts_delta.moved
                ));
            }
            if raw_delta.excused > 0 {
                excused_total += raw_delta.excused;
                excused_worst = excused_worst.max(raw_delta.worst_excused_margin);
            }
            if ts_delta.max_delta_ms > raw_delta.max_delta_ms {
                amplified.push(format!(
                    "{}: raw moved {} values (worst {}ms) -> reported worst {}ms",
                    c.clip, raw_delta.moved, raw_delta.max_delta_ms, ts_delta.max_delta_ms));
            }
            lang_raw_max = lang_raw_max.max(raw_delta.max_delta_ms);
            lang_ts_max = lang_ts_max.max(ts_delta.max_delta_ms);
        }

        if let Some(why) = unsupported {
            println!("{:<14} {:<11} {:>5} {:>7} {:>9} {:>10} {:>9} {:>8} {:>8}   NOT SUPPORTED: {}",
                     config, language, 0, 0, "-", "-", "-", "-", "-", why);
            covering.2 += 1;
            continue;
        }

        println!("{:<14} {:<11} {:>5} {:>7} {:>4}/{:<4} {:>7}ms {:>5}/{:<4} {:>6}ms {:>8.2}",
                 config, language, clips, words,
                 raw_exact, clips, lang_raw_max,
                 ts_exact, clips, lang_ts_max,
                 if elapsed_s > 0.0 { audio_s / elapsed_s } else { 0.0 });
        covering.0 += clips;
        covering.1 += 1;
        covering.3 += words;
        covering.4 += raw_exact;
        covering.5 += ts_exact;
        worst_raw = worst_raw.max(lang_raw_max);
        worst_ts = worst_ts.max(lang_ts_max);
    }

    println!();
    println!("=== {} languages, {} clips, {} words ===", covering.1, covering.0, covering.3);
    println!("    axis 2  raw argmax    : {}/{} clips exact, worst delta {}ms",
             covering.4, covering.0, worst_raw);
    println!("    axis 3  reported      : {}/{} clips exact, worst delta {}ms",
             covering.5, covering.0, worst_ts);
    if covering.6 == 0 {
        println!("    margin gate           : PASS \
                  (no endpoint moved where the reference's margin > {MARGIN_NOISE_FLOOR})");
    } else {
        println!("    margin gate           : FAIL on {} clip(s)", covering.6);
    }
    if excused_total > 0 {
        println!("    {excused_total} endpoint(s) excused below the noise floor \
                  (largest margin among them {excused_worst:.5})");
    }
    if probes > 0 {
        println!("    {probes} probe clip(s) run without a frozen reference");
    }
    if covering.2 > 0 {
        println!("    {} language(s) not covered by the port", covering.2);
    }
    if !amplified.is_empty() {
        println!("\n    {} clip(s) where `_fix_timestamps` turned a small raw delta into a bigger one:",
                 amplified.len());
        for a in amplified.iter().take(15) {
            println!("      {a}");
        }
    }
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    Ok(())
}

fn build_input_with(
    model_dir: &std::path::Path,
    ts_id: u32,
    words: &[String],
    valid: usize,
) -> Result<qwen3_aligner_wgpu::align_input::AlignerInput> {
    use std::cell::RefCell;
    thread_local! {
        static BUILDER: RefCell<Option<(std::path::PathBuf, u32, InputBuilder)>> =
            const { RefCell::new(None) };
    }
    BUILDER.with(|slot| {
        let mut b = slot.borrow_mut();
        let needs = !matches!(&*b, Some((p, t, _)) if p == model_dir && *t == ts_id);
        if needs {
            *b = Some((model_dir.to_path_buf(), ts_id, InputBuilder::load(model_dir, ts_id)?));
        }
        b.as_ref().unwrap().2.build(words, valid)
    })
}

/// The real thing: mel -> audio tower -> scatter -> text prefill -> timestamp
/// head -> repair, then a line-by-line diff against the frozen gold.
fn run(args: Vec<String>, model_dir: PathBuf, dtype: Dtype) -> Result<()> {
    use qwen3_aligner_wgpu::align_inference::Aligner;
    use qwen3_aligner_wgpu::gold::compare;
    use qwen3_aligner_wgpu::gpu::DeviceSelector;
    use qwen3_aligner_wgpu::postprocess::decode_timestamps;

    let spec = arg_value(&args, "--device").unwrap_or_else(|| "auto".to_string());
    let selector = DeviceSelector::parse(&spec)?;
    let fixtures = gold::fixtures_dir();

    let clips: Vec<String> = if args.iter().any(|a| a == "--all") {
        gold::CLIPS.iter().map(|s| s.to_string()).collect()
    } else {
        vec![arg_value(&args, "--wav").context("--wav <clip> or --all is required")?]
    };

    let t_load = Instant::now();
    let mut aligner = Aligner::load(selector, &model_dir)?;
    println!("device   : {}", aligner.describe());
    println!(
        "model    : {}  (labels={} segment={}ms)",
        model_dir.display(),
        aligner.config().num_labels,
        aligner.config().timestamp_segment_time_ms
    );
    println!("load     : {:.1}s", t_load.elapsed().as_secs_f64());
    println!();

    let builder = InputBuilder::load(&model_dir, aligner.config().timestamp_token_id)?;
    let sr = 16000u32;
    let mut failed = 0usize;
    let mut table = Vec::new();

    for clip in &clips {
        let gold = GoldJson::load(dtype, clip)?;
        let t0 = Instant::now();

        // Three separate phases on purpose.  `load_audio_wav` decodes *and*
        // resamples (soxr HQ, matching librosa's default), which for a 44.1 kHz
        // 176 s clip is seconds of work — folding it into "mel" hides the single
        // largest phase of the whole pipeline behind a 20 ms STFT.
        let t = Instant::now();
        let samples = qwen3_aligner_wgpu::load_audio_wav(fixtures.join(format!("{clip}.wav")), sr)?;
        let wav_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let (mel, _bins, _frames) = qwen3_aligner_wgpu::mel::mel_features(&samples)?;
        let valid = align_input::valid_mel_frames(samples.len());
        let padded = align_input::padded_mel_frames(valid, builder.n_window());
        let mut mel_padded = mel;
        mel_padded.resize(128 * padded, 0.0);
        let mel_ms = t.elapsed().as_secs_f64() * 1000.0;

        let words = words::split_words(&gold.transcript, Some(&gold.language))?;
        let input = builder.build(&words, valid)?;

        let raw_ms = aligner.align_raw_ms(&mel_padded, valid, &input)?;
        let items = decode_timestamps(&input.words, &raw_ms)?;
        let elapsed = t0.elapsed().as_secs_f64();

        let v = compare(&gold, dtype, &items, elapsed);
        let tm = aligner.timings;
        println!(
            "{:<9} wav={:>7.1} mel={:>5.1} enc={:>7.1} gather={:>5.1} prefill={:>8.1} head={:>7.1} ms",
            clip, wav_ms, mel_ms, tm.enc_ms, tm.gather_ms, tm.prefill_ms, tm.head_ms
        );
        println!(
            "          RTFx {:.2}  elapsed {:.3}s",
            gold.audio_seconds / elapsed,
            elapsed
        );
        println!("          {}", v.summary());
        if !v.passes(1) {
            failed += 1;
            // Show the words that actually moved, not the first six (which are
            // usually all correct and tell you nothing).
            let mut shown = 0usize;
            for (i, (a, b)) in items.iter().zip(&gold.items).enumerate() {
                if (a.start_time - b.start_time).abs() < 1e-9
                    && (a.end_time - b.end_time).abs() < 1e-9
                {
                    continue;
                }
                println!(
                    "            [{i:>4}] {:<14} ours {:.3}..{:.3}   gold {:.3}..{:.3}   \
                     (start {:+}ms, end {:+}ms)",
                    a.text,
                    a.start_time,
                    a.end_time,
                    b.start_time,
                    b.end_time,
                    ((a.start_time - b.start_time) * 1000.0).round() as i64,
                    ((a.end_time - b.end_time) * 1000.0).round() as i64,
                );
                shown += 1;
                if shown >= 12 {
                    break;
                }
            }
        }
        table.push(v);
        println!();
    }

    println!("=== {} clip(s), {failed} not passing ===", table.len());
    if failed > 0 {
        bail!("{failed} clip(s) did not match the gold");
    }
    Ok(())
}

const USAGE: &str = "\
align — Qwen3-ForcedAligner, wgpu port

USAGE:
  align --wav <clip>                     align one fixture, diff against the gold
  align --all                            all six fixtures
  align --check-input --all              gate only the input path (no model)
  align --list-clips                     show the covered clips

OPTIONS:
  --model <dir>    checkpoint dir        (default: QALIGN_MODEL or the -hf download)
  --device <spec>  auto | cpu | vulkan | dx12 | metal | gl | <adapter name substring>
  --dtype <t>      f16 | bf16 | fp32     (default: f16)
";
