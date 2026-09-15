#!/usr/bin/env python3
"""Dump Python -hf intermediates for alignment (mel, encoder hidden, token ids)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf")
    p.add_argument("--wav", default=r"D:\qwen3-asr-rs\tests\fixtures\90s_en.wav")
    p.add_argument("--out", default=r"D:\qwen3-asr-wgpu\align_dump\py_90s_en")
    p.add_argument("--max-new", type=int, default=1024)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda:0", low_cpu_mem_usage=True
    )
    model.eval()

    inputs = processor.apply_transcription_request(audio=args.wav)
    inputs = inputs.to(model.device, torch.float16)

    feats = inputs["input_features"].detach().float().cpu().numpy()
    np.save(out / "input_features.npy", feats)
    print("input_features", feats.shape, feats.dtype, "min", feats.min(), "max", feats.max(), "mean", feats.mean())

    if "input_features_mask" in inputs:
        mask = inputs["input_features_mask"].detach().cpu().numpy()
        np.save(out / "input_features_mask.npy", mask)
        print("mask", mask.shape, "sum", mask.sum())

    ids = inputs["input_ids"].detach().cpu().numpy()
    np.save(out / "input_ids.npy", ids)
    print("input_ids", ids.shape, "len", ids.shape[-1])

    with torch.inference_mode():
        # encoder hidden via model.audio_tower if present
        audio_out = None
        feats = inputs["input_features"]
        mask = inputs.get("input_features_mask")
        mm = getattr(model, "model", model)
        enc = getattr(mm, "audio_tower", None)
        proj = getattr(mm, "multi_modal_projector", None)
        print("encoder", type(enc), "proj", type(proj))
        if enc is not None:
            raw = enc(feats, mask) if mask is not None else enc(feats)
            if hasattr(raw, "last_hidden_state"):
                raw = raw.last_hidden_state
            ah0 = raw.detach().float().cpu().numpy()
            np.save(out / "encoder_hidden.npy", ah0)
            print("encoder_hidden", ah0.shape, "min", ah0.min(), "max", ah0.max(), "mean", ah0.mean())
            if proj is not None:
                raw = proj(raw)
            audio_out = raw
        if audio_out is not None:
            if hasattr(audio_out, "last_hidden_state"):
                audio_out = audio_out.last_hidden_state
            ah = audio_out.detach().float().cpu().numpy()
            np.save(out / "audio_embeds.npy", ah)
            print("audio_embeds", ah.shape, "min", ah.min(), "max", ah.max(), "mean", ah.mean())

        out_ids = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        gen = out_ids[:, inputs["input_ids"].shape[1] :]
        np.save(out / "gen_ids.npy", gen.cpu().numpy())
        parsed = processor.decode(gen, return_format="parsed")[0]
        print("lang", parsed.get("language"))
        print("text", parsed.get("transcription"))
        (out / "text.txt").write_text(parsed.get("transcription") or "", encoding="utf-8")

    print("wrote", out)


if __name__ == "__main__":
    main()
