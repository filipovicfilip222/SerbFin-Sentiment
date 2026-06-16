"""
Raw -> clean JSONL pipeline (data/raw/ -> data/interim/clean.jsonl).

Steps applied in order:
  1. Strip known site-name suffixes from text  (e.g. " | Biznis.rs")
  2. Normalize whitespace
  3. Drop rows outside sane word-count band [MIN_WORDS, MAX_WORDS]
  4. Drop rows matching junk patterns
  5. Re-dedup on normalized text (case-insensitive)
  6. Re-assign sequential sfs-NNNNNN IDs from 1

Outputs data/interim/clean.jsonl by default.
Reports: total dropped per reason.  Does NOT label anything.

Usage:
    python -m src.data.clean
    python -m src.data.clean --input data/raw/headlines.jsonl --output data/interim/clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.data.inspect import _FIN_CUE  # reuse the cue-word pattern for the final heuristic

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_WORDS = 5
MAX_WORDS = 30

# Site-name suffix patterns: strip these from the END of the text.
# Order matters: try longer/more-specific patterns first.
_SUFFIX_PATTERNS: list[re.Pattern] = [
    re.compile(r"\s*[|\-]\s*Biznis\.rs\s*$",           re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*novaekonomija\.rs\s*$",    re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*Nova Ekonomija\s*$",       re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*B92\.net\s*$",             re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*B92\s*$",                  re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*eKapija(?:\.com)?\s*$",    re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*Kamatica(?:\.com)?\s*$",   re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*Blic(?:\.rs)?\s*$",        re.IGNORECASE),
    re.compile(r"\s*[|\-]\s*Politika(?:\.rs)?\s*$",    re.IGNORECASE),
    # Generic catch-all: "anything - SiteName.tld" where tld in rs/com/net/org
    re.compile(r"\s*[|\-]\s*\w[\w\-]*\.(rs|com|net|org)\s*$", re.IGNORECASE),
]

# Junk patterns (applied to lowercased stripped text after suffix removal).
# These identify navigation fragments, error pages, and consent dialogs.
_JUNK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("navigation",    re.compile(r"^\s*najnovije\s*[|\-\s]")),
    ("404",           re.compile(r"\b404\b")),
    ("page_not_found", re.compile(r"\bpage not found\b")),
    ("cookie_consent", re.compile(
        r"\b(kolači[cć]|cookie|prihvati sve|prihvati kola|privacy policy|politika privat)\b"
    )),
    ("bare_domain",   re.compile(r"^[a-z0-9\s\.\-]+\.(rs|com|net|org)\s*$")),
]

DropReason = str  # one of: "too_short", "too_long", "junk:<pattern>", "duplicate"


# ── Text normalisation ─────────────────────────────────────────────────────────

def _strip_suffix(text: str) -> str:
    for pat in _SUFFIX_PATTERNS:
        cleaned = pat.sub("", text).strip()
        if cleaned != text:
            return cleaned  # only apply the first matching pattern
    return text


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(text: str) -> str:
    return _normalize_ws(_strip_suffix(text))


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# ── Drop checks ───────────────────────────────────────────────────────────────

def _drop_reason(text: str) -> DropReason | None:
    wc = len(text.split())
    if wc < MIN_WORDS:
        return "too_short"
    if wc > MAX_WORDS:
        return "too_long"
    lower = text.lower()
    for label, pat in _JUNK_PATTERNS:
        if pat.search(lower):
            return f"junk:{label}"
    return None


# ── Pipeline ──────────────────────────────────────────────────────────────────

def clean_file(
    input_path: Path,
    output_path: Path,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Run the full cleaning pipeline.
    Returns a dict of drop-reason -> count plus "kept" and "total".
    """
    with input_path.open(encoding="utf-8") as fh:
        raw_lines = [l.strip() for l in fh if l.strip()]

    drop_counts: dict[str, int] = {}
    kept: list[dict] = []
    seen_norm: set[str] = set()

    for lineno, line in enumerate(raw_lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            reason = "json_error"
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
            if verbose:
                print(f"  [drop] line {lineno}: {reason} -- {exc}", file=sys.stderr)
            continue

        original = rec.get("text", "")
        text = _clean_text(original)
        rec["text"] = text  # update in-place

        reason = _drop_reason(text)
        if reason:
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
            if verbose:
                print(f"  [drop] {reason:20s}  {repr(original[:70])}")
            continue

        norm = _normalize_for_dedup(text)
        if norm in seen_norm:
            reason = "duplicate"
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
            if verbose:
                print(f"  [drop] {reason:20s}  {repr(text[:70])}")
            continue

        seen_norm.add(norm)
        kept.append(rec)

    # Re-assign sequential IDs from 1
    for i, rec in enumerate(kept, 1):
        rec["id"] = f"sfs-{i:06d}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    drop_counts["kept"] = len(kept)
    drop_counts["total"] = len(raw_lines)
    return drop_counts


def print_report(stats: dict[str, int], output_path: Path) -> None:
    total  = stats.get("total", 0)
    kept   = stats.get("kept", 0)
    drops  = {k: v for k, v in stats.items() if k not in ("kept", "total")}
    n_drop = total - kept

    print("=" * 64)
    print(f"Clean pipeline results")
    print("=" * 64)
    print(f"  Input records  : {total:>6}")
    print(f"  Dropped total  : {n_drop:>6}  ({n_drop/max(total,1):.1%})")
    print()
    print("  Drop breakdown:")
    for reason, n in sorted(drops.items(), key=lambda x: -x[1]):
        pct = n / max(total, 1) * 100
        print(f"    {reason:<22}  {n:>5}  ({pct:.1f}%)")
    print()
    print(f"  Kept (clean)   : {kept:>6}  ({kept/max(total,1):.1%})")
    print(f"  Output         : {output_path}")
    print()

    # Per-source breakdown of kept records
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            kept_recs = [json.loads(l) for l in fh if l.strip()]
        from collections import Counter
        per_src = Counter(r.get("source", "?") for r in kept_recs)
        print("  Per-source (clean):")
        for src, n in sorted(per_src.items(), key=lambda x: -x[1]):
            print(f"    {src:<30} {n:>5}")
        print()

        # Finance-cue heuristic
        texts = [r.get("text", "") for r in kept_recs]
        with_cue    = sum(1 for t in texts if _FIN_CUE.search(t))
        without_cue = len(texts) - with_cue
        print("  Finance/sentiment cue-word heuristic (relevance signal, not a label):")
        print(f"    With finance cue    : {with_cue:>5}  ({with_cue/max(len(texts),1):.1%})")
        print(f"    Without finance cue : {without_cue:>5}  ({without_cue/max(len(texts),1):.1%})")
        print("    Cue words checked: profit, gubitak, rast, pad, dobit, otpuštanja,")
        print("    rekord, kriza, prihod, investicij, akcij, berz, kamat, BDP, ...")
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Clean raw SerbFin headlines -> data/interim/clean.jsonl"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/headlines.jsonl"),
        help="Raw JSONL file (default: data/raw/headlines.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/clean.jsonl"),
        help="Output path (default: data/interim/clean.jsonl)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each dropped row",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    print(f"Cleaning {args.input} ...")
    stats = clean_file(args.input, args.output, verbose=args.verbose)
    print_report(stats, args.output)


if __name__ == "__main__":
    main()
