#!/usr/bin/env python3
"""Python-HF reference hypotheses for the FLEURS sweep (one model, one wav at a time).

Same protocol as `qwen3-asr-rs/scripts/freeze_python_hf_cuda_f16.py` (fp16 on
cuda:0, greedy, `apply_transcription_request`), but driven by
`tools/fleurs_langs.tsv` + the extracted FLEURS test split, so the output drops
straight into `tools/score_asr.py`.

    python tools/run_python_hf.py --model D:\\Qwen3-ASR\\models\\Qwen3-ASR-0.6B-hf \
        --limit 200 --out eval_data/fleurs

Writes `eval_data/fleurs/<config>.py.hyps.tsv` (`filename<TAB>text`) — the same
format `src/bin/eval_asr.rs --hyps-out` produces, so the two are comparable
per utterance.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def read_langs(tsv: Path) -> list[tuple[str, str]]:
    """`(config, primary_metric)` for every language the table lists."""
    out = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) >= 3 and cols[1] != "--":
            out.append((cols[1], cols[2]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path("eval_data/fleurs"))
    ap.add_argument("--limit", type=int, default=200, help="clips per language (head of the tsv)")
    ap.add_argument("--even", type=int, default=0, help="clips per language, evenly spaced by file size")
    ap.add_argument("--langs", default="", help="comma-separated configs (default: all in the table)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tag", default="py")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1
    root = a.root
    if a.langs:
        metrics = dict(read_langs(Path(__file__).resolve().parent / "fleurs_langs.tsv"))
        configs = [(c, metrics.get(c, "wer")) for c in a.langs.split(",")]
    else:
        configs = read_langs(Path(__file__).resolve().parent / "fleurs_langs.tsv")

    print(f"GPU: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}  fp16", flush=True)
    print(f"model: {a.model}", flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(a.model))
    model = AutoModelForMultimodalLM.from_pretrained(
        str(a.model), dtype=torch.float16, device_map="cuda:0", low_cpu_mem_usage=True
    )
    model.eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s", flush=True)

    for cfg, metric in configs:
        tsv = root / "tsv" / f"{cfg}.test.tsv"
        wav_dir = root / "audio" / cfg / "test"
        if not tsv.is_file() or not wav_dir.is_dir():
            print(f"SKIP {cfg} (missing data)", flush=True)
            continue
        rows = []
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) >= 4:
                rows.append((cols[1], cols[3]))
        if a.even:
            # Same clip set as the wgpu runner: sorted by file size (16 kHz mono
            # PCM16 everywhere, so size order is duration order), evenly spaced.
            rows.sort(key=lambda r: ((wav_dir / r[0]).stat().st_size if (wav_dir / r[0]).is_file() else 0, r[0]))
            step = len(rows) / a.even
            rows = [rows[int(i * step)] for i in range(a.even) if int(i * step) < len(rows)]
        else:
            rows = rows[: a.limit]

        out = root / f"{cfg}.{a.tag}.hyps.tsv"
        lang_out = root / f"{cfg}.{a.tag}.langs.tsv"
        summary_out = root / f"{cfg}.{a.tag}.summary.tsv"
        done = set()
        mode = "w"
        if out.is_file():  # resume a partial sweep
            for line in out.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    done.add(line.split("\t", 1)[0])
            mode = "a"
            print(f"  {cfg}: resuming, {len(done)} already done", flush=True)
        lang_mode = "a" if done and lang_out.is_file() else "w"

        elapsed_s = 0.0
        audio_s = 0.0
        n_empty = 0
        langs: dict[str, int] = {}
        with out.open(mode, encoding="utf-8") as f, lang_out.open(lang_mode, encoding="utf-8") as fl:
            for i, (name, _ref) in enumerate(rows):
                if name in done:
                    continue
                wav = wav_dir / name
                if not wav.is_file():
                    continue
                # "read the wav -> text", the same span the wgpu runner times.
                t1 = time.perf_counter()
                inputs = processor.apply_transcription_request(audio=str(wav))
                inputs = inputs.to(model.device, torch.float16)
                plen = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t1
                parsed = processor.decode(ids[:, plen:], return_format="parsed")[0]
                text = (parsed.get("transcription") or "").strip()
                language = (parsed.get("language") or "").strip()
                text = text.replace("\n", " ").replace("\r", " ")
                f.write(f"{name}\t{text}\n")
                f.flush()
                fl.write(f"{name}\t{language}\n")
                fl.flush()
                elapsed_s += dt
                audio_s += sf.info(str(wav)).frames / sf.info(str(wav)).samplerate
                langs[language or "-"] = langs.get(language or "-", 0) + 1
                if not text:
                    n_empty += 1
                if (i + 1) % 25 == 0:
                    print(f"  {cfg}: {i + 1}/{len(rows)}  {dt:.2f}s/clip  empty={n_empty}", flush=True)
        rtfx = audio_s / elapsed_s if elapsed_s > 0 else 0.0
        summary_out.write_text(
            "system\tconfig\tclips\taudio_s\telapsed_s\trtfx\tload_s\tmetric\tcorpus_pct\tmean_pct\tunits\tsub\tdel\tins\tlanguages\n"
            f"python\t{cfg}\t{len(rows)}\t{audio_s:.3f}\t{elapsed_s:.3f}\t{rtfx:.3f}\t{load_s:.1f}\t{metric}\t"
            f"nan\tnan\t0\t0\t0\t0\t{' '.join(f'{k}={v}' for k, v in langs.items())}\n",
            encoding="utf-8",
        )
        print(
            f"[done] {cfg}: {len(rows) - len(done)} clips, {elapsed_s:.1f}s  audio {audio_s:.1f}s  "
            f"RTFx {rtfx:.2f}  empty={n_empty}",
            flush=True,
        )

    print("PYTHON_HF_FLEURS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
