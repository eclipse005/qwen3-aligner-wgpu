#!/usr/bin/env python3
"""Is a cross-system text difference a near-tie?

Runs the Python reference on one clip with `output_scores=True`, so every greedy
step carries its full logit vector, and prints the steps where the top-1 margin
is small — i.e. the places where a one-ulp difference between two
implementations can pick the other token.

    python tools/logit_margin.py --model D:\\Qwen3-ASR\\models\\Qwen3-ASR-0.6B-hf ^
        --wav eval_data\\fleurs\\audio\\da_dk\\test\\10444405129133538025.wav --top 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--wav", required=True, type=Path)
    ap.add_argument("--top", type=int, default=5, help="candidates printed around each near-tie")
    ap.add_argument("--threshold", type=float, default=0.5, help="print steps whose top-1 margin is below this")
    ap.add_argument("--all", action="store_true", help="print every step, not just the near-ties")
    ap.add_argument("--max-new", type=int, default=256)
    a = ap.parse_args()

    processor = AutoProcessor.from_pretrained(str(a.model))
    model = AutoModelForMultimodalLM.from_pretrained(
        str(a.model), dtype=torch.float16, device_map="cuda:0", low_cpu_mem_usage=True
    ).eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else None

    inputs = processor.apply_transcription_request(audio=str(a.wav))
    inputs = inputs.to(model.device, torch.float16)
    plen = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=a.max_new, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    ids = out.sequences[0, plen:].tolist()
    scores = torch.stack(out.scores)  # [steps, vocab]

    def tok_text(t: int) -> str:
        if tokenizer is None:
            return str(t)
        return tokenizer.decode([t], skip_special_tokens=False)

    margins = []
    for step in range(scores.shape[0]):
        lg = scores[step].reshape(-1).float()  # generate keeps a batch dim
        top = torch.topk(lg, a.top)
        m = (top.values[0] - top.values[1]).item()
        margins.append(m)
        if a.all or m <= a.threshold:
            cands = "  |  ".join(
                f"{tok_text(int(t))!r} {v:+.3f}" for t, v in zip(top.indices, top.values)
            )
            tag = "  <<< near-tie" if m <= a.threshold else ""
            print(f"step {step:>3}  chose {tok_text(ids[step])!r}  margin {m:.4f}{tag}\n        {cands}")

    margins.sort()
    if margins:
        n = len(margins)
        print(
            f"\n{m} steps   min margin {margins[0]:.4f}   "
            f"p10 {margins[n // 10]:.3f}   median {margins[n // 2]:.3f}   "
            f"below {a.threshold}: {sum(1 for x in margins if x <= a.threshold)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
