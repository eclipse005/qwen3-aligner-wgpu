#!/usr/bin/env python
"""Classify the per-clip differences between two systems' FLEURS hypotheses.

Alignment being < 100% is only alarming if the clips differ *a lot*: a text that
matches except for a comma is numeric jitter at an f16 boundary, while a text
that diverges wholesale is a porting bug.  This classifies every differing clip
into

  cosmetic  — identical after normalization (punctuation/case/space only)
  1-2 units — one or two words/characters
  3+ units  — worth reading by hand (printed in full)

    python tools/diff_systems.py --ours 0p6 --theirs 0p6 --tag-suffix ...
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=HERE.parent / "eval_data" / "fleurs")
    ap.add_argument("--ours", default="w06")
    ap.add_argument("--other", default="py06")
    ap.add_argument("--langs", default="", help="comma-separated configs")
    ap.add_argument("--show", type=int, default=2, help="how many 3+ clips to print per language")
    a = ap.parse_args()

    want = set(a.langs.split(",")) if a.langs else None
    print(f"{'language':<12} {'clips':>5} {'ident':>5} {'cosm':>5} {'1-2u':>5} {'3+u':>5}  worst")
    tot = {"clips": 0, "ident": 0, "cosm": 0, "small": 0, "big": 0}
    for lang_name, cfg, metric, code in read_langs():
        if want is not None and cfg not in want:
            continue
        tsv = a.root / "tsv" / f"{cfg}.test.tsv"
        ours = read_dump(a.root / f"{cfg}.{a.ours}.hyps.tsv")
        theirs = read_dump(a.root / f"{cfg}.{a.other}.hyps.tsv")
        if not tsv.is_file() or not ours or not theirs:
            continue
        refs = read_pairs(tsv)
        strip = code in ("zh", "ja")
        counts = {"ident": 0, "cosm": 0, "small": 0, "big": 0}
        big_rows = []
        for key, our_text in ours.items():
            if key not in theirs:
                continue
            tot["clips"] += 1
            counts_any = True
            our_norm = normalize(our_text, code, strip)
            their_norm = normalize(theirs[key], code, strip)
            if our_text.strip() == theirs[key].strip():
                counts["ident"] += 1
                continue
            if our_norm == their_norm:
                counts["cosm"] += 1
                continue
            ua = units(our_norm, metric)
            ub = units(their_norm, metric)
            e = sum(edit_counts(ub, ua))
            if e <= 2:
                counts["small"] += 1
            else:
                counts["big"] += 1
                big_rows.append((e, len(ub), key, refs.get(key, ""), theirs[key], our_text))
        for k in counts:
            tot[k] += counts[k]
        worst = f"{counts['big']} clip(s) with 3+ units" if counts["big"] else ""
        print(
            f"{lang_name:<12} {sum(counts.values()):>5} {counts['ident']:>5} {counts['cosm']:>5} "
            f"{counts['small']:>5} {counts['big']:>5}  {worst}"
        )
        for e, n, key, ref, theirs_t, our_t in sorted(big_rows, reverse=True)[: a.show]:
            print(f"    err {e}/{n}  {key}")
            print(f"      ref  : {ref[:130]}")
            print(f"      {a.other}: {theirs_t[:130]}")
            print(f"      {a.ours}: {our_t[:130]}")
    print(
        f"\nTOTAL clips {tot['clips']}  identical {tot['ident']} ({100 * tot['ident'] / max(tot['clips'], 1):.1f}%)  "
        f"cosmetic {tot['cosm']}  1-2u {tot['small']}  3+u {tot['big']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
