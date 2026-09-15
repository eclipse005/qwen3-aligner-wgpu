//! `align` — forced alignment, and the gate that keeps it honest.
//!
//! ```text
//! align --audio speech.wav --text transcript.txt --language English
//! align gate --device vulkan --dtype fp32
//! ```
//!
//! `gate` is a separate subcommand because it needs the frozen gold, the six
//! fixtures and the FLEURS tree, which a user of the aligner does not have.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use qwen3_aligner_wgpu::align_inference::Aligner;
use qwen3_aligner_wgpu::gold::{self, Dtype, GoldJson};
use qwen3_aligner_wgpu::gpu::DeviceSelector;
use qwen3_aligner_wgpu::postprocess::AlignItem;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-devices") {
        for d in pollster::block_on(qwen3_aligner_wgpu::list_devices()) {
            println!("{}", d.describe());
        }
        return Ok(());
    }

    let model_dir = arg_value(&args, "--model")
        .map(PathBuf::from)
        .unwrap_or_else(gold::model_dir);

    // Name the path and say what is wrong with it.  Left to the loader this
    // surfaces as `os error 3` from several frames down, which says nothing
    // about which directory was meant or what belongs in it.
    if !model_dir.is_dir() {
        bail!(
            "model directory not found: {}\n  \
             pass --model <dir>, or set QALIGN_MODEL, or put the checkpoint at \
             the default location",
            model_dir.display()
        );
    }
    if !model_dir.join("config.json").is_file() {
        bail!(
            "{} does not look like a checkpoint: no config.json in it\n  \
             expected the `-hf` one (architectures = Qwen3ASRForTokenClassification, \
             model.audio_tower.* / model.language_model.* / score.weight)",
            model_dir.display()
        );
    }

    let device = arg_value(&args, "--device").unwrap_or_else(|| "auto".to_string());
    let selector = DeviceSelector::parse(&device)?;

    // The subcommand is the first positional — never a flag's *value*, which is
    // how `--audio speech.wav` would otherwise be read as a subcommand named
    // after the file.
    match args.first().map(|s| s.as_str()) {
        Some("gate") => return gate(&args, &model_dir, selector),
        Some(s) if !s.starts_with('-') => {
            bail!("unknown subcommand {s:?} (did you mean `gate`?)")
        }
        _ => {}
    }
    align(&args, &model_dir, selector)
}

/// The reference's call, from the command line.
fn align(args: &[String], model_dir: &Path, selector: DeviceSelector) -> Result<()> {
    let audio = arg_value(args, "--audio").context("--audio <wav> is required")?;
    let audio = PathBuf::from(audio);
    anyhow::ensure!(audio.is_file(), "no such audio file: {}", audio.display());

    // An existing path is read; anything else is the transcript itself.  The
    // reference takes the text inline, and handing it a transcript file is the
    // usual awkwardness that comes with that — this accepts both.
    let text_arg = arg_value(args, "--text").context("--text <file|string> is required")?;
    let text = if Path::new(&text_arg).is_file() {
        std::fs::read_to_string(&text_arg).with_context(|| text_arg.clone())?
    } else {
        text_arg
    };
    let text = text.trim().to_string();
    anyhow::ensure!(!text.is_empty(), "--text is empty");

    let language = arg_value(args, "--language");

    let t_load = Instant::now();
    let mut aligner = Aligner::load(selector, model_dir)?;
    eprintln!(
        "device {}  load {:.1}s  languages {}",
        aligner.describe(),
        t_load.elapsed().as_secs_f64(),
        aligner.supported_languages().len()
    );

    let t0 = Instant::now();
    let items = aligner.align(&audio, &text, language.as_deref())?;
    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(out) = arg_value(args, "--output") {
        let json = serde_json::to_string_pretty(
            &items
                .iter()
                .map(|i| {
                    serde_json::json!({
                        "text": i.text,
                        "start_time": i.start_time,
                        "end_time": i.end_time
                    })
                })
                .collect::<Vec<_>>(),
        )?;
        std::fs::write(&out, json).with_context(|| format!("write {out}"))?;
        eprintln!("wrote {out} ({} items)", items.len());
    } else {
        print_items(&items);
    }

    let tm = aligner.timings;
    eprintln!(
        "{} items in {:.3}s  (wav+mel+decode {:.1} / enc {:.1} / gather {:.1} / prefill {:.1} / head {:.1} ms)",
        items.len(),
        elapsed,
        // `total_ms` is timed *inside* the forward, so what it does not cover is
        // the wav, the mel, the word split and the pairing.
        elapsed * 1000.0 - tm.total_ms,
        tm.enc_ms,
        tm.gather_ms,
        tm.prefill_ms,
        tm.head_ms,
    );
    Ok(())
}

fn print_items(items: &[AlignItem]) {
    for it in items {
        println!("{}\t{:.3}\t{:.3}", it.text, it.start_time, it.end_time);
    }
}

// ---------------------------------------------------------------------------
// gate
// ---------------------------------------------------------------------------

/// The regression gate: the frozen gold, the six fixtures, the FLEURS sample.
fn gate(args: &[String], model_dir: &Path, selector: DeviceSelector) -> Result<()> {
    let dtype = arg_value(args, "--dtype")
        .map(|s| Dtype::parse(&s).unwrap_or_else(|| panic!("unknown --dtype {s}")))
        .unwrap_or(Dtype::Fp32);

    if args.iter().any(|a| a == "--check-input") {
        return gate_input(model_dir);
    }
    if args.iter().any(|a| a == "--eval") {
        return gate_eval(args, model_dir, selector);
    }
    gate_fixtures(args, model_dir, selector, dtype)
}

/// wav -> words -> `input_ids`, with no model in the loop.
fn gate_input(model_dir: &Path) -> Result<()> {
    use qwen3_aligner_wgpu::align_input::{valid_mel_frames, InputBuilder};
    use qwen3_aligner_wgpu::words;

    let cfg = qwen3_aligner_wgpu::config::AsrConfig::from_file(&model_dir.join("config.json"))?;
    let head = cfg
        .align
        .as_ref()
        .context("checkpoint has no forced-aligner head")?;
    let builder = InputBuilder::load(model_dir, head.timestamp_token_id as u32)?;
    let fixtures = gold::fixtures_dir();

    let mut bad = 0usize;
    for clip in gold::CLIPS {
        let g = GoldJson::load(Dtype::F16, clip)?;
        let samples =
            qwen3_aligner_wgpu::load_audio_wav(fixtures.join(format!("{clip}.wav")), 16000)?;
        let valid = valid_mel_frames(samples.len());
        let w = words::split_words(&g.transcript, Some(&g.language))?;
        let input = builder.build(&w, valid)?;
        let ok = w == g.words
            && input.input_ids == g.input_ids
            && input.n_audio_tokens == g.n_audio_tokens
            && input.padded_mel_frames == g.mel_frames;
        if !ok {
            bad += 1;
        }
        println!(
            "{clip:<9} {:<8} words={:<4} seq={:<5} audio={:<5} mel={}",
            if ok { "OK" } else { "MISMATCH" },
            w.len(),
            input.seq_len(),
            input.n_audio_tokens,
            input.padded_mel_frames
        );
    }
    anyhow::ensure!(bad == 0, "{bad} clip(s) did not match");
    println!("\nall clips match the reference input exactly");
    Ok(())
}

/// Run the model over the six fixtures and diff against the frozen gold.
fn gate_fixtures(
    args: &[String],
    model_dir: &Path,
    selector: DeviceSelector,
    dtype: Dtype,
) -> Result<()> {
    let fixtures = gold::fixtures_dir();
    let clips: Vec<String> = match arg_value(args, "--wav") {
        Some(c) => vec![c],
        None => gold::CLIPS.iter().map(|s| s.to_string()).collect(),
    };

    let t_load = Instant::now();
    let mut aligner = Aligner::load(selector, model_dir)?;
    println!("device : {}", aligner.describe());
    println!("gold   : {}/{}", gold::gold_root().display(), dtype.dir_name());
    println!("load   : {:.1}s\n", t_load.elapsed().as_secs_f64());

    let mut failed = 0usize;
    for clip in &clips {
        let g = GoldJson::load(dtype, clip)?;
        let audio = fixtures.join(format!("{clip}.wav"));
        let t0 = Instant::now();
        let items = aligner.align(&audio, &g.transcript, Some(&g.language))?;
        let elapsed = t0.elapsed().as_secs_f64();

        let v = gold::compare(&g, dtype, &items, elapsed);
        let tm = aligner.timings;
        println!(
            "{clip:<9} outer={:>6.1} enc={:>7.1} gather={:>5.1} prefill={:>8.1} head={:>6.1} ms",
            elapsed * 1000.0 - tm.total_ms,
            tm.enc_ms,
            tm.gather_ms,
            tm.prefill_ms,
            tm.head_ms,
        );
        println!(
            "          RTFx {:.2}  elapsed {:.3}s",
            g.audio_seconds / elapsed,
            elapsed
        );
        println!("          {}", v.summary());
        if !v.passes(1) {
            failed += 1;
            // Show the words that actually moved, not the first six (which are
            // usually all correct and tell you nothing).
            for (i, (a, b)) in items.iter().zip(&g.items).enumerate() {
                if (a.start_time - b.start_time).abs() < 1e-9
                    && (a.end_time - b.end_time).abs() < 1e-9
                {
                    continue;
                }
                println!(
                    "            [{i:>4}] {:<14} ours {:.3}..{:.3}   gold {:.3}..{:.3}",
                    a.text, a.start_time, a.end_time, b.start_time, b.end_time
                );
            }
        }
        println!();
    }
    println!("=== {} clip(s), {failed} not passing ===", clips.len());
    anyhow::ensure!(failed == 0, "{failed} clip(s) did not match the gold");
    Ok(())
}

/// One clip of the multi-language FLEURS eval.
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

/// The multi-language sweep, judged by the margin gate.
fn gate_eval(args: &[String], model_dir: &Path, selector: DeviceSelector) -> Result<()> {
    use qwen3_aligner_wgpu::gold::{TsVerdict, MARGIN_NOISE_FLOOR};

    let root = gold::gold_root().parent().unwrap().join("eval");
    let only: Vec<String> = arg_value(args, "--langs")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let limit: Option<usize> = arg_value(args, "--limit").and_then(|s| s.parse().ok());

    let mut aligner = Aligner::load(selector, model_dir)?;
    println!("device : {}", aligner.describe());
    println!("eval   : {}\n", root.display());
    println!(
        "{:<14} {:<11} {:>5} {:>7} {:>9} {:>10} {:>9} {:>8} {:>8}",
        "config", "language", "clips", "words", "raw-exact", "raw-maxdelta", "ts-exact",
        "ts-maxdelta", "RTFx"
    );

    // (languages, clips, unsupported, words, raw-exact, ts-exact, gate failures)
    let mut sum = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut worst_raw = 0i64;
    let mut worst_ts = 0i64;
    let mut excused = 0usize;
    let mut excused_worst = 0.0f32;
    let mut probes = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .with_context(|| format!("read {}", root.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    dirs.sort();

    for dir in dirs {
        let config = dir.file_name().unwrap().to_string_lossy().to_string();
        if !only.is_empty() && !only.contains(&config) {
            continue;
        }
        let mut files: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .map(|e| e.path())
            .collect();
        files.sort();
        // The ASR round's methodology: a fixed sample per language, not the whole
        // split.  The full FLEURS gold is on disk and can be spot-checked later,
        // but verifying all of it costs an hour of GPU to re-derive what 20 clips
        // per language already establishes.
        if let Some(n) = limit {
            files.truncate(n);
        }

        let (mut clips, mut words) = (0usize, 0usize);
        let (mut raw_exact, mut ts_exact, mut bad) = (0usize, 0usize, 0usize);
        let (mut lang_raw, mut lang_ts) = (0i64, 0i64);
        let (mut audio_s, mut elapsed_s) = (0.0f64, 0.0f64);
        let mut language = String::new();
        let mut unsupported: Option<String> = None;

        for path in files {
            let c: EvalClip = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
            language = c.language.clone();
            clips += 1;
            audio_s += c.audio_seconds;

            let t0 = Instant::now();
            // A language the tokeniser cannot handle yet is a gap in coverage,
            // not a clip failure: report it rather than aborting the run and
            // hiding the languages that do work.
            let (raw_ms, items) =
                match aligner.align_with_raw(Path::new(&c.wav), &c.transcript, Some(&c.language)) {
                    Ok(v) => v,
                    Err(e) => {
                        unsupported = Some(e.to_string());
                        break;
                    }
                };
            elapsed_s += t0.elapsed().as_secs_f64();
            words += c.words.len();

            if items.len() != c.words.len()
                || items.iter().zip(&c.words).any(|(i, w)| &i.text != w)
            {
                bad += 1;
                failures.push(format!("{}: word list differs", c.clip));
                continue;
            }
            // A clip with no frozen reference is a *probe*: the question there is
            // whether the port completes at all, and counting an empty comparison
            // as "exact" would manufacture agreement.
            if c.raw_masked_ms.is_empty() {
                probes += 1;
                println!(
                    "      probe {:<16} words={:<5} items={}",
                    c.clip,
                    c.words.len(),
                    items.len()
                );
                continue;
            }

            // Two axes, and they are not the same axis.  `raw_ms` is what the
            // towers produced and is where `margin[i]` is meaningful; the items
            // are that stream after `_fix_timestamps`, which can move a value in
            // from elsewhere.  The gate is on the raw one.
            let raw_v = TsVerdict::of(&raw_ms, &c.raw_masked_ms, &c.margin);
            let ts: Vec<i64> = items
                .iter()
                .flat_map(|i| {
                    [
                        (i.start_time * 1000.0).round() as i64,
                        (i.end_time * 1000.0).round() as i64,
                    ]
                })
                .collect();
            let ts = TsVerdict::of(&ts, &c.fixed_ms, &c.margin);
            if raw_v.moved == 0 {
                raw_exact += 1;
            }
            if ts.moved == 0 {
                ts_exact += 1;
            }
            if raw_v.beyond_floor > 0 {
                bad += 1;
                failures.push(format!(
                    "{}: {} raw endpoint(s) moved at a confident position (margin > {}), worst {}ms",
                    c.clip, raw_v.beyond_floor, MARGIN_NOISE_FLOOR, raw_v.max_delta_ms
                ));
            }
            // `fix_timestamps` is deterministic, so identical raw input must give
            // identical output.  If that ever fails, the repair is reading
            // something outside its argument.
            if ts.moved > 0 && raw_v.moved == 0 {
                failures.push(format!(
                    "{}: raw matches everywhere but {} reported endpoint(s) moved — \
                     the repair is not a function of the raw values alone",
                    c.clip, ts.moved
                ));
            }
            excused += raw_v.excused + ts.excused;
            excused_worst = excused_worst
                .max(raw_v.worst_excused_margin)
                .max(ts.worst_excused_margin);
            lang_raw = lang_raw.max(raw_v.max_delta_ms);
            lang_ts = lang_ts.max(ts.max_delta_ms);
        }

        if let Some(why) = unsupported {
            println!("{config:<14} {language:<11}   NOT SUPPORTED: {why}");
            sum.2 += 1;
            continue;
        }

        println!(
            "{:<14} {:<11} {:>5} {:>7} {:>4}/{:<4} {:>7}ms {:>4}/{:<4} {:>6}ms {:>8.2}",
            config,
            language,
            clips,
            words,
            raw_exact,
            clips,
            lang_raw,
            ts_exact,
            clips,
            lang_ts,
            if elapsed_s > 0.0 { audio_s / elapsed_s } else { 0.0 }
        );
        sum.0 += clips;
        sum.1 += 1;
        sum.3 += words;
        sum.4 += raw_exact;
        sum.5 += ts_exact;
        sum.6 += bad;
        worst_raw = worst_raw.max(lang_raw);
        worst_ts = worst_ts.max(lang_ts);
    }

    println!();
    println!("=== {} languages, {} clips, {} words ===", sum.1, sum.0, sum.3);
    println!("    raw argmax : {}/{} clips exact, worst {worst_raw}ms", sum.4, sum.0);
    println!("    reported   : {}/{} clips exact, worst {worst_ts}ms", sum.5, sum.0);
    if sum.6 == 0 {
        println!(
            "    margin gate: PASS (no endpoint moved where the reference's margin > {MARGIN_NOISE_FLOOR})"
        );
    } else {
        println!("    margin gate: FAIL on {} clip(s)", sum.6);
    }
    if excused > 0 {
        println!("    {excused} endpoint(s) excused below the noise floor (worst margin {excused_worst:.5})");
    }
    if probes > 0 {
        println!("    {probes} probe clip(s) run without a frozen reference");
    }
    if sum.2 > 0 {
        println!("    {} language(s) not covered by the port", sum.2);
    }
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    anyhow::ensure!(sum.6 == 0, "{} clip(s) failed the margin gate", sum.6);
    Ok(())
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

const USAGE: &str = "\
align — Qwen3-ForcedAligner, Rust + wgpu

USAGE:
  align --audio <wav> --text <file|string> --language <name> [--output <json>]
        Align one clip.  An existing path for --text is read as a file, anything
        else is the transcript itself.  Without --output the items are printed as
        `word<TAB>start<TAB>end`.

  align gate [--wav <clip>] [--dtype <t>] [--device <spec>]
        The model over the six fixtures, diffed against the frozen gold.

  align gate --eval [--langs <a,b>] [--limit <n>] [--device <spec>]
        The multi-language FLEURS sweep, under the margin gate.

  align gate --check-input
        The input path only (wav -> words -> input_ids); no model needed.

  align --list-devices
        Every wgpu adapter, with the limits that decide which kernels run.

OPTIONS:
  --model <dir>    checkpoint directory.  Default: `QALIGN_MODEL` if set, else
                   D:\\Qwen3-ASR\\models\\Qwen3-ForcedAligner-0.6B-tf.
  --device <spec>  auto | cpu | vulkan | dx12 | metal | gl | <adapter substring>
  --dtype <t>      fp32 (default) | f16 | bf16 — which gold `gate` compares against

NOTE:
  The aligner always stores f16 and accumulates in f32, so --dtype is a gate-time
  choice, not a compute knob.  fp32 is the default because the reference agrees
  with itself exactly at fp32, and disagrees with fp16 at a handful of near-tie
  positions.
";
