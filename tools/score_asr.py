#!/usr/bin/env python
"""Corpus WER/CER with standards-compliant text normalization.

Scores the hypotheses dumped by `src/bin/eval_asr.rs --hyps-out` against the
references of a FLEURS `test.tsv` (2-column `filename<TAB>reference` manifests
work too).

Normalization, both sides, in this order:
  * whisper's `EnglishTextNormalizer` for `--lang en` — the de-facto standard,
    which collapses "twentieth century" and "20th-century" and turns "two" into
    "2" so that the model's digit style is not scored as an error;
  * `cn2an` (optional, `--lang zh`) so "二十九岁" and "29 岁" agree;
  * whisper's `BasicTextNormalizer` (lowercase, symbols out, CJK-aware spacing)
    for everything else.
Whitespace is then dropped for `--mode cer`.

The edit distance is computed here (Levenshtein with the S/D/I breakdown) and
cross-checked against `jiwer` when it is importable.

```text
python tools/score_asr.py --tsv eval_data/fleurs/tsv/en_us.test.tsv ^
    --hyps eval_data/fleurs/en_us.hyps.tsv --mode wer --lang en --worst 15
```
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

try:
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer

    _EN = EnglishTextNormalizer()
    _BASIC = BasicTextNormalizer()
    _BASIC_NODIAC = BasicTextNormalizer(remove_diacritics=True)
except ImportError:  # pragma: no cover - the fallback keeps the tool usable
    _EN = _BASIC = _BASIC_NODIAC = None

# Arabic/Persian: diacritics and tatweel are optional in writing, and alef/ya/
# ta-marbuta spellings vary between the model and the reference, so fold them
# before scoring (same fold on both sides).
_AR_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
_AR_FOLD = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    }
)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cn2an

    def _zh_numbers(text: str) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return cn2an.transform(text, "cn2an")
            except Exception:
                return text

except ImportError:  # pragma: no cover
    def _zh_numbers(text: str) -> str:
        return text

try:
    import jiwer
except ImportError:  # pragma: no cover
    jiwer = None


def normalize(text: str, lang: str, strip_latin: bool = False) -> str:
    if strip_latin:
        # FLEURS zh/ja keep the original Latin name in the reference
        # (`…埃尔多安recep tayyip erdoğan通话…`) although the recording only
        # carries the transliteration, so scoring it as an error is measuring
        # the dataset, not the model.  Latin/other-script runs are dropped on
        # both sides so the two views stay comparable.
        text = re.sub(r"[^\u3000-\u9fff\uff00-\uffef]+", " ", text) if lang in ("zh", "ja") else text
    if lang in ("ar", "fa"):
        text = _AR_MARKS.sub("", text).translate(_AR_FOLD)
    if _EN is None:  # no whisper-normalizer: keep it plain but symmetric
        text = text.lower()
        text = "".join(c if c.isalnum() else " " for c in text)
    elif lang == "en":
        text = _EN(text)
    elif lang in ("ar", "fa"):
        text = _BASIC_NODIAC(text)
    else:
        text = _BASIC(text)
    if lang == "zh":
        text = _zh_numbers(text)
    return re.sub(r"\s+", " ", text).strip()


def units(text: str, mode: str) -> list[str]:
    if mode == "wer":
        return text.split()
    return [c for c in text if not c.isspace()]


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein with the substitution/deletion/insertion breakdown."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )
    i, j = n, m
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def read_pairs(tsv: Path) -> dict[str, str]:
    """`filename -> reference` from a FLEURS tsv or a 2-column manifest."""
    pairs: dict[str, str] = {}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) >= 4:
            pairs[cols[1]] = cols[3]
        elif len(cols) == 2:
            pairs[cols[0]] = cols[1]
        else:
            sys.exit(f"{tsv}: row with {len(cols)} columns")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--hyps", required=True, type=Path, help="filename<TAB>text from eval_asr")
    ap.add_argument("--mode", choices=("wer", "cer"), default="wer")
    ap.add_argument("--lang", default="", help="en, zh, ja, ... (picks the normalizer)")
    ap.add_argument(
        "--strip-latin",
        action="store_true",
        help="drop Latin-script runs from both sides (FLEURS zh/ja references carry ones that were never spoken)",
    )
    ap.add_argument("--worst", type=int, default=0, help="print the N worst utterances")
    ap.add_argument("--out-tsv", type=Path, help="write `file<TAB>rate<TAB>ref<TAB>hyp`")
    ap.add_argument(
        "--only",
        type=Path,
        help="score only the clips present in this `file<TAB>text` dump (same clip set as the other system)",
    )
    ap.add_argument(
        "--compare",
        type=Path,
        help="another `file<TAB>text` dump (e.g. the Python-HF run): report agreement + distance",
    )
    a = ap.parse_args()

    refs = read_pairs(a.tsv)
    if a.only:
        keep = set()
        for line in a.only.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                keep.add(line.split("\t", 1)[0])
        refs = {k: v for k, v in refs.items() if k in keep}
    hyps = {}
    for line in a.hyps.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            hyps[k] = v

    rows, missing = [], 0
    for key, ref in refs.items():
        if key not in hyps:
            missing += 1
            continue
        r = units(normalize(ref, a.lang, a.strip_latin), a.mode)
        h = units(normalize(hyps[key], a.lang, a.strip_latin), a.mode)
        sub, dele, ins = edit_counts(r, h)
        errs = sub + dele + ins
        rate = errs / len(r) if r else 0.0
        rows.append((key, len(r), errs, rate, ref, hyps[key], r, h))

    if not rows:
        sys.exit("no overlapping rows between the tsv and the hypothesis dump")
    total_units = sum(r[1] for r in rows)
    total_errors = sum(r[2] for r in rows)
    corpus = total_errors / total_units
    mean = sum(r[3] for r in rows) / len(rows)
    median = sorted(r[3] for r in rows)[len(rows) // 2]

    print(f"clips {len(rows)}" + (f"  (missing {missing})" if missing else ""))
    print(f"corpus {100 * corpus:6.2f}%   mean/utt {100 * mean:6.2f}%   median/utt {100 * median:6.2f}%")
    print(f"units {total_units}   errors {total_errors}")

    if jiwer is not None:
        # Cross-check the aggregate with an independent implementation: the units
        # we scored are whitespace-separated either way (words for WER, single
        # characters for CER), so jiwer's word-level rate is the same metric.
        truth = [" ".join(r[6]) for r in rows]
        pred = [" ".join(r[7]) for r in rows]
        j = jiwer.wer(truth, pred)
        print(f"jiwer cross-check: {100 * j:6.2f}%   (ours {100 * corpus:6.2f}%)")

    if a.out_tsv:
        with a.out_tsv.open("w", encoding="utf-8") as f:
            for key, _n, _e, rate, ref, hyp, _r, _h in rows:
                f.write(f"{key}\t{rate:.4f}\t{ref}\t{hyp}\n")

    if a.compare:
        other = {}
        for line in a.compare.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                other[k] = v
        same = total = 0
        dist_units = dist_errs = 0
        for key, hyp in ((r[0], r[5]) for r in rows):
            if key not in other:
                continue
            total += 1
            if normalize(hyp, a.lang, a.strip_latin) == normalize(other[key], a.lang, a.strip_latin):
                same += 1
            o = units(normalize(other[key], a.lang, a.strip_latin), a.mode)
            u = units(normalize(hyp, a.lang, a.strip_latin), a.mode)
            sub, dele, ins = edit_counts(o, u)
            dist_units += len(o)
            dist_errs += sub + dele + ins
        if total:
            print(
                f"vs {a.compare.name}: identical {same}/{total} ({100 * same / total:.1f}%), "
                f"distance {100 * dist_errs / max(dist_units, 1):.2f}% over {dist_units} units"
            )
        else:
            print(f"vs {a.compare.name}: no overlapping clips")

    if a.worst:
        print(f"\n--- {a.worst} worst ---")
        for key, n, errs, rate, ref, hyp, _r, _h in sorted(rows, key=lambda r: -r[3])[: a.worst]:
            print(f"\n{key}  ref{n} err{errs} {100 * rate:5.1f}%")
            print(f"  ref: {ref}")
            print(f"  hyp: {hyp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
