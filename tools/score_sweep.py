#!/usr/bin/env python
"""Score an FLEURS sweep and print a markdown table (ours vs the Python-HF run).

    python tools/score_sweep.py --tag 0p6 --py-tag py --limit 200 ^
        --out docs/eval-fleurs-0p6.md

Per language it reports the primary metric from `tools/fleurs_langs.tsv` (CER for
the scripts without word boundaries, WER otherwise), how a second system scored
on the same clips, the delta, and how often the two systems produced the *same*
text — the parity view, which is a sharper bug detector than either WER alone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def read_langs() -> list[tuple[str, str, str, str]]:
    rows = []
    for line in (HERE / "fleurs_langs.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) >= 3 and cols[1] != "--":
            code = cols[3] if len(cols) > 3 else cols[0][:2].lower()
            rows.append((cols[0], cols[1], cols[2], code))
    return rows


def score(tsv: Path, hyps: Path, mode: str, code: str, py: Path | None, only: Path | None = None) -> dict:
    cmd = [sys.executable, str(HERE / "score_asr.py"), "--tsv", str(tsv), "--hyps", str(hyps),
           "--mode", mode, "--lang", code]
    if code in ("zh", "ja"):
        cmd.append("--strip-latin")
    # Restrict both systems to the clips they have in common, otherwise a partial
    # reference dump would be compared against a longer run.
    if only is not None:
        cmd += ["--only", str(only)]
    if py is not None:
        cmd += ["--compare", str(py)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
    res = {"raw": out}
    for line in out.splitlines():
        if line.startswith("corpus"):
            res["corpus"] = float(line.split()[1].rstrip("%"))
            parts = line.split()
            res["mean"] = float(parts[parts.index("mean/utt") + 1].rstrip("%"))
        elif line.startswith("clips"):
            res["clips"] = int(line.split()[1])
        elif line.startswith("vs "):
            res["same"] = line.split("identical")[1].split("(")[1].split("%")[0]
            res["distance"] = float(line.split("distance")[1].split("%")[0])
    return res


def read_langs_dump(path: Path) -> dict[str, str]:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                out[k] = v.strip()
    return out


def lid_accuracy(dump: dict[str, str], expected: str) -> tuple[int, int, dict[str, int]]:
    """(correct, total, confusions) for one language's detected-language dump."""
    correct = 0
    confusions: dict[str, int] = {}
    for got in dump.values():
        if got == expected:
            correct += 1
        else:
            confusions[got or "<empty>"] = confusions.get(got or "<empty>", 0) + 1
    return correct, len(dump), confusions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="0p6")
    ap.add_argument("--py-tag", default="py")
    ap.add_argument("--root", type=Path, default=REPO / "eval_data" / "fleurs")
    ap.add_argument("--langs", default="", help="comma-separated configs")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    want = set(a.langs.split(",")) if a.langs else None
    table = [r for r in read_langs() if want is None or r[1] in want]

    lines = ["| language | config | metric | clips | ours | python | delta | identical | LID ours | LID py |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    ours_all, py_all = [], []
    lid_totals = [0, 0]
    lid_py_totals = [0, 0]
    confusions: list[str] = []
    for lang_name, cfg, metric, code in table:
        tsv = a.root / "tsv" / f"{cfg}.test.tsv"
        hyps = a.root / f"{cfg}.{a.tag}.hyps.tsv"
        pyhs = a.root / f"{cfg}.{a.py_tag}.hyps.tsv"
        if not hyps.is_file():
            continue
        r = score(tsv, hyps, metric, code, pyhs if pyhs.is_file() else None,
                  only=pyhs if pyhs.is_file() else None)
        p = score(tsv, pyhs, metric, code, None, only=hyps) if pyhs.is_file() else None
        ours = r.get("corpus", float("nan"))
        pyv = p.get("corpus") if p else None
        delta = f"{pyv - ours:+.2f}" if pyv is not None else "-"
        identical = r.get("same", "-")
        ours_all.append(ours)
        if pyv is not None:
            py_all.append(pyv)

        correct, total, wrong = lid_accuracy(read_langs_dump(a.root / f"{cfg}.{a.tag}.langs.tsv"), lang_name)
        lid_totals[0] += correct
        lid_totals[1] += total
        lid_ours = f"{100 * correct / total:.1f}%" if total else "-"
        if wrong:
            confusions.append(f"  {lang_name} ({cfg}): " + ", ".join(f"{k}×{v}" for k, v in sorted(wrong.items())))

        py_dump = read_langs_dump(a.root / f"{cfg}.{a.py_tag}.langs.tsv")
        pc, pt, _ = lid_accuracy(py_dump, lang_name)
        lid_py_totals[0] += pc
        lid_py_totals[1] += pt
        lid_py = f"{100 * pc / pt:.1f}%" if pt else "-"

        lines.append(
            f"| {lang_name} | {cfg} | {metric.upper()} | {r.get('clips', 0)} | "
            f"**{ours:.2f}%** | {f'{pyv:.2f}%' if pyv is not None else '-'} | {delta} | "
            f"{identical}% | {lid_ours} | {lid_py} |"
        )
        print(lines[-1], flush=True)

    if ours_all:
        mean_ours = sum(ours_all) / len(ours_all)
        tail = f"\nlanguages {len(ours_all)}   mean {mean_ours:.2f}%"
        if py_all:
            mean_py = sum(py_all) / len(py_all)
            tail += f"   python {mean_py:.2f}%   delta {mean_py - mean_ours:+.2f}"
        if lid_totals[1]:
            tail += f"   LID {100 * lid_totals[0] / lid_totals[1]:.1f}%"
            if lid_py_totals[1]:
                tail += f" (python {100 * lid_py_totals[0] / lid_py_totals[1]:.1f}%)"
        lines.append(tail)
        print(tail)
    if confusions:
        lines += ["", "LID confusions:"] + confusions
        print("\nLID confusions:")
        print("\n".join(confusions))

    if a.out:
        a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
