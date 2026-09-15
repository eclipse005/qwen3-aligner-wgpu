//! `align` — forced alignment from the command line.
//!
//! ```text
//! align --audio speech.wav --text transcript.txt --language English
//! align --audio speech.wav --text "hello world" --language English --output out.json
//! align --list-devices
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use qwen3_aligner_wgpu::align_inference::Aligner;
use qwen3_aligner_wgpu::gpu::DeviceSelector;
use qwen3_aligner_wgpu::paths;
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
        .unwrap_or_else(paths::model_dir);
    if !model_dir.is_dir() {
        bail!(
            "model directory not found: {}\n  \
             pass --model <dir>, or set QALIGN_MODEL",
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

    let audio = arg_value(&args, "--audio").context("--audio <wav> is required")?;
    let audio = PathBuf::from(audio);
    anyhow::ensure!(audio.is_file(), "no such audio file: {}", audio.display());

    // An existing path is read; anything else is the transcript itself.
    let text_arg = arg_value(&args, "--text").context("--text <file|string> is required")?;
    let text = if Path::new(&text_arg).is_file() {
        std::fs::read_to_string(&text_arg).with_context(|| text_arg.clone())?
    } else {
        text_arg
    };
    let text = text.trim().to_string();
    anyhow::ensure!(!text.is_empty(), "--text is empty");

    let language = arg_value(&args, "--language");

    let t_load = Instant::now();
    let mut aligner = Aligner::load(selector, &model_dir)?;
    eprintln!(
        "device {}  load {:.1}s",
        aligner.describe(),
        t_load.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let items = aligner.align(&audio, &text, language.as_deref())?;
    let elapsed = t0.elapsed().as_secs_f64();

    match arg_value(&args, "--output") {
        Some(out) => {
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
        }
        None => print_items(&items),
    }

    let tm = aligner.timings;
    eprintln!(
        "{} items in {:.3}s  (wav+mel+decode {:.1} / enc {:.1} / gather {:.1} / prefill {:.1} / head {:.1} ms)",
        items.len(),
        elapsed,
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

  align --list-devices
        Every wgpu adapter, with the limits that decide which kernels run.

OPTIONS:
  --model <dir>    checkpoint directory; default `QALIGN_MODEL`, else
                   D:\\Qwen3-ASR\\models\\Qwen3-ForcedAligner-0.6B-tf.
  --device <spec>  auto | cpu | vulkan | dx12 | metal | gl | <adapter substring>
";
