#!/usr/bin/env python
"""The final FLEURS table: WER/CER + RTFx + per-clip alignment, all four runs.

Per language it reports

  * WER/CER of the port and of the Python reference, for 0.6B *and* 1.7B
    (four numbers, the primary metric of that language from `fleurs_langs.tsv`),
  * RTFx of all four runs (same timing span: "read the wav -> text", one GPU job
    at a time),
  * how often the port and Python produced byte-identical text on the same clip
    (alignment %, plus the unit-level distance between the two systems).

    python tools/score_final.py --tags w06=0p6,w17=1p7,py06=py,py17=py17 \
        --out docs/eval-fleurs-final.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_asr import edit_counts, normalize, read_pairs, units  # noqa: E402


def read_langs() -> list[tuple[str, str, str, str]]:
    rows = []
    for line in (HERE / "fleurs_langs.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) >= 3 and cols[1] != "--":
            rows.append((cols[0], cols[1], cols[2], cols[3] if len(cols) > 3 else "en"))
    return rows


def read_dump(path: Path) -> dict[str, str]:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                out[k] = v
    return out


def read_rtfx(path: Path) -> float | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None
    head, row = lines[0].split("\t"), lines[1].split("\t")
    try:
        return float(row[head.index("rtfx")])
    except (ValueError, IndexError):
        return None


def score(refs: dict[str, str], hyps: dict[str, str], mode: str, code: str) -> tuple[float, int] | None:
    strip = code in ("zh", "ja")
    total_units = total_errs = clips = 0
    for key, ref in refs.items():
        if key not in hyps:
            continue
        r = units(normalize(ref, code, strip), mode)
        h = units(normalize(hyps[key], code, strip), mode)
        total_errs += sum(edit_counts(r, h))
        total_units += len(r)
        clips += 1
    if not clips or not total_units:
        return None
    return 100.0 * total_errs / total_units, clips


def align(a: dict[str, str], b: dict[str, str], mode: str, code: str) -> tuple[float, float, int] | None:
    """(identical %, unit distance %, clips) between two systems on the same clips."""
    strip = code in ("zh", "ja")
    same = total = dist_units = dist_errs = 0
    for key, ta in a.items():
        if key not in b:
            continue
        total += 1
        if normalize(ta, code, strip) == normalize(b[key], code, strip):
            same += 1
        ua = units(normalize(ta, code, strip), mode)
        ub = units(normalize(b[key], code, strip), mode)
        dist_errs += sum(edit_counts(ua, ub))
        dist_units += len(ua)
    if not total:
        return None
    return 100.0 * same / total, 100.0 * dist_errs / max(dist_units, 1), total


def fmt(v: float | None, suffix: str = "%") -> str:
    return "-" if v is None else f"{v:.2f}{suffix}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tags",
        default="w06=0p6,w17=1p7,py06=py,py17=py17",
        help="name=tag pairs for the four runs",
    )
    ap.add_argument("--root", type=Path, default=HERE.parent / "eval_data" / "fleurs")
    ap.add_argument("--langs", default="", help="comma-separated configs")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    tags = dict(p.split("=", 1) for p in a.tags.split(","))
    order = list(tags)  # w06, w17, py06, py17
    want = set(a.langs.split(",")) if a.langs else None

    header = (
        "| language | metric | clips | wgpu 0.6B | python 0.6B | delta | wgpu 1.7B | python 1.7B | delta "
        "| align 0.6B | align 1.7B | RTFx w0.6 | py0.6 | w1.7 | py1.7 |"
    )
    sep = "|" + "---|" * (header.count("|") - 1)
    lines = [header, sep]
    totals: dict[str, list[float]] = {k: [] for k in order}
    aligns: dict[str, list[float]] = {"w06": [], "w17": []}

    for lang_name, cfg, metric, code in read_langs():
        if want is not None and cfg not in want:
            continue
        tsv = a.root / "tsv" / f"{cfg}.test.tsv"
        if not tsv.is_file():
            continue
        refs = read_pairs(tsv)
        hyps = {k: read_dump(a.root / f"{cfg}.{t}.hyps.tsv") for k, t in tags.items()}
        scores = {k: score(refs, hyps[k], metric, code) for k in order}
        if all(v is None for v in scores.values()):
            continue
        clips = max((v[1] for v in scores.values() if v), default=0)
        rtfx = {k: read_rtfx(a.root / f"{cfg}.{t}.summary.tsv") for k, t in tags.items()}
        al06 = align(hyps["w06"], hyps["py06"], metric, code)
        al17 = align(hyps["w17"], hyps["py17"], metric, code)

        def pct(k: str) -> float | None:
            return scores[k][0] if scores[k] else None

        def delta(ours: float | None, py: float | None) -> str:
            if ours is None or py is None:
                return "-"
            return f"{py - ours:+.2f}"

        for k in order:
            if scores[k]:
                totals[k].append(scores[k][0])
        for name, al in (("w06", al06), ("w17", al17)):
            if al:
                aligns[name].append(al[0])

        lines.append(
            f"| {lang_name} | {metric.upper()} | {clips} | **{fmt(pct('w06'))}** | {fmt(pct('py06'))} "
            f"| {delta(pct('w06'), pct('py06'))} | **{fmt(pct('w17'))}** | {fmt(pct('py17'))} "
            f"| {delta(pct('w17'), pct('py17'))} | {fmt(al06[0]) if al06 else '-'} | {fmt(al17[0]) if al17 else '-'} "
            f"| {fmt(rtfx['w06'], 'x')} | {fmt(rtfx['py06'], 'x')} | {fmt(rtfx['w17'], 'x')} | {fmt(rtfx['py17'], 'x')} |"
        )
        print(lines[-1], flush=True)

    def mean(v: list[float]) -> str:
        return f"{sum(v) / len(v):.2f}%" if v else "-"

    lines.append("")
    lines.append(
        f"languages {len(totals['w06']) or len(totals['w17'])}   "
        f"mean: wgpu0.6 {mean(totals['w06'])}  py0.6 {mean(totals['py06'])}  "
        f"wgpu1.7 {mean(totals['w17'])}  py1.7 {mean(totals['py17'])}   "
        f"alignment: 0.6B {mean(aligns['w06'])}  1.7B {mean(aligns['w17'])}"
    )
    print("\n" + lines[-1])

    if a.out:
        a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
