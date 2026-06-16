"""
Data-quality inspection report for any SerbFin-Sentiment JSONL file.
Read-only: prints statistics, never mutates.

Usage:
    python -m src.data.inspect --input data/raw/headlines.jsonl
    python -m src.data.inspect --input data/interim/clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── Configurable bad-pattern list ──────────────────────────────────────────────
# Patterns are applied to the *lowercased* stripped text AFTER suffix removal.
# Add entries here; each is a compiled regex.
_BAD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("navigation",   re.compile(r"^\s*najnovije\s*[|\-]")),
    ("404",          re.compile(r"\b404\b")),
    ("page_not_found", re.compile(r"\bpage not found\b")),
    ("cookie_consent", re.compile(r"\b(kolači[cć]|cookie|prihvati sve|prihvati kola|privacy policy|politika privat)\b")),
    ("just_sitename", re.compile(r"^[a-z0-9\s\.\-]+\.(rs|com|net|org)\s*$")),
]

# ── Known site-name suffix fragments (for suffix detection report only) ────────
_SUFFIX_RE = re.compile(
    r"[\s\-|]+("
    r"biznis\.rs|b92\.net|b92|nova ekonomija|novaekonomija\.rs|"
    r"ekapija\.com|ekapija|kamatica\.com|kamatica|"
    r"blic\.rs|blic|politika\.rs|politika"
    r")\s*$",
    re.IGNORECASE,
)

# ── Finance / sentiment cue words (heuristic relevance only, not a label) ─────
_FIN_CUE = re.compile(
    r"\b("
    r"profit|gubitak|rast|pad|dobit|otpu[sš]tanj|otpu[sš]ti|rekord|kriza|"
    r"prihod|rashod|investicij|dug|dolar|evro|dinar|berz|akcij|obveznc|"
    r"obveznic|kamat|bdp|inflacij|penzij|deficit|suficit|subvencij|"
    r"kreditn|bankrot|likvidacij|ste[cč]aj|restrukturir|dividend|tr[žz]i[šs]t|"
    r"uvoz|izvoz|deviz|forint|ruble|kurs|porast|smanjenje|pove[cć]anje|"
    r"zarad|plat|zaposlen|nezaposlen|zaduž"
    r")\w*",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split())


def _load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: JSON error -- {exc}", file=sys.stderr)
    return records


def run_report(path: Path) -> None:
    records = _load(path)
    if not records:
        print("No records found.")
        return

    texts = [r.get("text", "") for r in records]
    total = len(records)

    print("=" * 64)
    print(f"File     : {path}")
    print(f"Records  : {total:,}")
    print("=" * 64)

    # ── Per-source counts ──────────────────────────────────────────────────────
    src_counts = Counter(r.get("source", "?") for r in records)
    print("\nPer-source counts:")
    for src, n in sorted(src_counts.items(), key=lambda x: -x[1]):
        bar = "#" * int(n / total * 40)
        print(f"  {src:<30} {n:>5}  {bar}")

    # ── Date range + year histogram ────────────────────────────────────────────
    dates = sorted(r.get("date", "") for r in records if r.get("date"))
    if dates:
        print(f"\nDate range: {dates[0]}  ->  {dates[-1]}")
    year_hist = Counter(d[:4] for d in dates if len(d) >= 4)
    print("Year histogram:")
    for yr in sorted(year_hist, reverse=True):
        bar = "#" * int(year_hist[yr] / total * 40)
        print(f"  {yr}  {year_hist[yr]:>5}  {bar}")

    # ── Word-count band ────────────────────────────────────────────────────────
    wc_list = [_word_count(t) for t in texts]
    short = [(i, texts[i]) for i, wc in enumerate(wc_list) if wc < 5]
    long_ = [(i, texts[i]) for i, wc in enumerate(wc_list) if wc > 30]
    print(f"\nWord-count: min={min(wc_list)}, max={max(wc_list)}, "
          f"median={sorted(wc_list)[total // 2]}")
    print(f"  Short (<5 words)  : {len(short):>5}")
    for _, t in short[:8]:
        print(f"    {repr(t[:80])}")
    if len(short) > 8:
        print(f"    ... ({len(short) - 8} more)")
    print(f"  Long  (>30 words) : {len(long_):>5}")
    for _, t in long_[:5]:
        print(f"    {repr(t[:100])}")

    # ── Site-name suffixes still present ──────────────────────────────────────
    with_suffix = [t for t in texts if _SUFFIX_RE.search(t)]
    print(f"\nSite-name suffixes still present: {len(with_suffix):>5} / {total}")
    for t in with_suffix[:5]:
        print(f"  {repr(t[:100])}")
    if len(with_suffix) > 5:
        print(f"  ... ({len(with_suffix) - 5} more)")

    # ── Junk-pattern matches ───────────────────────────────────────────────────
    print("\nJunk-pattern matches:")
    total_junk = 0
    for label, pat in _BAD_PATTERNS:
        hits = [t for t in texts if pat.search(t.lower())]
        total_junk += len(hits)
        print(f"  {label:<20} : {len(hits):>4}")
        for t in hits[:3]:
            print(f"    {repr(t[:90])}")
    print(f"  Total junk        : {total_junk:>4}")

    # ── Near-duplicate texts (normalized) ─────────────────────────────────────
    normed = [re.sub(r"\s+", " ", t.lower().strip()) for t in texts]
    freq = Counter(normed)
    dupes = [(t, n) for t, n in freq.most_common(30) if n > 1]
    print(f"\nDuplicate normalized texts (count > 1): {len(dupes)}")
    for t, n in dupes[:20]:
        print(f"  {n}x  {repr(t[:80])}")

    # ── Finance-cue heuristic ──────────────────────────────────────────────────
    with_cue    = sum(1 for t in texts if _FIN_CUE.search(t))
    without_cue = total - with_cue
    print(f"\nFinance/sentiment cue-word heuristic (relevance signal only):")
    print(f"  With cue word    : {with_cue:>5}  ({with_cue/total:.1%})")
    print(f"  Without cue word : {without_cue:>5}  ({without_cue/total:.1%})")
    print("  (This is a rough relevance signal -- not a label or filter)")
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Print data-quality report for a SerbFin JSONL file"
    )
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"File not found: {args.input}")

    run_report(args.input)


if __name__ == "__main__":
    main()
