"""
Silver labeling: calls Claude Haiku with the EXACT prompt template from SPEC.md §5.1
to assign financial sentiment labels to raw JSONL records.

Usage:
    python -m src.label.silver \
        --input data/interim/clean.jsonl \
        --output data/interim/silver.jsonl \
        --resume
    # Safer defaults for Tier 1 (50 req/min org limit):
    python -m src.label.silver ... --workers 2 --rpm 45

Reads ANTHROPIC_API_KEY from environment (or .env file).

IMPORTANT: Use this only for the TRAIN pool. The gold test set must be human-labeled.
See SPEC.md §4.4 and data/ANNOTATION.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from src.data.schema import Label, LabelSource, SentimentRecord

load_dotenv()

# ── Prompt templates ─────────────────────────────────────────────────────────
#
# SYSTEM_PROMPT_V1 — original 3-class (SPEC.md §5.1).
#   Used at TRAIN / ZERO-SHOT BASELINE / EVAL time.
#   The trained model learns exactly this prompt.
#
# SYSTEM_PROMPT — 4-class with "irrelevant" and trap examples.
#   Used only for silver-v2 DATA LABELING / FILTERING.
#   "irrelevant" rows are stripped BEFORE training; the model stays 3-class.
#
# Keeping v1 allows the zero-shot baseline to use the exact same 3-class prompt
# as the fine-tuned model, making the comparison fair.

SYSTEM_PROMPT_V1 = (
    "Ti si klasifikator finansijskog sentimenta. "
    "Za dati naslov vesti odredi sentiment sa perspektive investitora.\n"
    "Odgovori isključivo jednom rečju: positive, negative ili neutral."
)

SYSTEM_PROMPT = (
    "Ti si klasifikator finansijskog sentimenta. "
    "Proceni da li naslov nosi direktan investitorski/tržišni signal i koji je.\n\n"
    "Odgovori ISKLJUČIVO jednom rečju: positive, negative, neutral ili irrelevant\n\n"
    "positive   = bullish signal (profit/prihod/BDP raste, nova investicija, "
                "beat expectations, rating upgrade, novi ugovor, M&A po premijumu)\n"
    "negative   = bearish signal (gubitak, bankrot, rating downgrade, "
                "miss expectations, troškovni pritisak koji urušava profitabilnost)\n"
    "neutral    = finansijska vest bez pravca (imenovanje, mešoviti rezultati, "
                "regulatorno saopštenje, plan bez ishoda, statistički izveštaj)\n"
    "irrelevant = nema direktnog tržišnog signala: politika, građanski/radnički "
                "protest, kulturni/sportski događaj, sajam bez finansijske objave, "
                "PR demant, opšte društvene vesti\n\n"
    "Ključne zamke:\n"
    "  Otpuštanja radi marže ili restrukturiranja → positive ili neutral  (NE negative)\n"
    "  'Radnici protestuju' / 'Meštani blokiraju' → irrelevant\n"
    "  'Kompanija počasni gost sajma' / nagrada bez fin. sadržaja → irrelevant\n"
    "  Rast cena energije / goriva → negative (troškovni pritisak na privredu)\n"
    "  Pad inflacije / pad kamatnih stopa → positive"
)

VALID_LABELS = {lbl.value for lbl in Label}
MODEL = "claude-haiku-4-5-20251001"


def build_user_message(text: str) -> str:
    return f"Naslov: {text}\nSentiment:"


# ── Thread-safe token-bucket rate limiter ────────────────────────────────────

class _RateLimiter:
    """
    Token-bucket rate limiter shared by all labeling threads.

    Allows at most `rpm` API calls per 60 s regardless of worker count.
    A small burst allowance lets the run start without stalling.
    """

    def __init__(self, rpm: int, burst: int = 3) -> None:
        self._rate = rpm / 60.0          # tokens / second
        self._tokens = float(burst)
        self._capacity = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one request token is available."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
            else:
                wait = (1.0 - self._tokens) / self._rate
                time.sleep(wait)
                self._tokens = 0.0
                self._last = time.monotonic()


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_label(raw: str) -> tuple[Optional[str], bool]:
    """
    Extract a valid label from the model's raw response.

    Returns (label, is_clean) where is_clean=False signals a malformed response
    that required regex fallback. Malformed rate is itself a result metric --
    it should approach 0 after fine-tuning vs higher for zero-shot.
    """
    clean = raw.strip().lower()
    if clean in VALID_LABELS:
        return clean, True
    match = re.search(r"\b(positive|negative|neutral|irrelevant)\b", clean)
    if match:
        return match.group(1), False
    return None, False


def _retry_after(exc: anthropic.RateLimitError) -> float:
    """Extract Retry-After seconds from a 429 response, or 0 if absent."""
    try:
        v = exc.response.headers.get("retry-after") or \
            exc.response.headers.get("x-ratelimit-reset-requests")
        return float(v) if v else 0.0
    except Exception:
        return 0.0


# ── API call with manual retry and Retry-After header support ─────────────────

def call_api(
    client: anthropic.Anthropic,
    text: str,
    rate_limiter: _RateLimiter,
    max_attempts: int = 6,
) -> tuple[Optional[str], bool]:
    """
    Call Claude Haiku and return (label, is_clean).
    Returns (None, False) after exhausting retries -- never raises.
    Rate-limited via shared token bucket; respects Retry-After header.
    """
    for attempt in range(max_attempts):
        rate_limiter.acquire()
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=10,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(text)}],
            )
            raw = response.content[0].text if response.content else ""
            return parse_label(raw)

        except anthropic.RateLimitError as exc:
            if attempt == max_attempts - 1:
                return None, False
            # Prefer the header value; fall back to exponential backoff
            wait = _retry_after(exc) or min(5.0 * (2 ** attempt), 90.0)
            time.sleep(wait)

        except anthropic.APIStatusError as exc:
            if exc.status_code in (500, 502, 503, 529):
                if attempt == max_attempts - 1:
                    return None, False
                time.sleep(min(5.0 * (2 ** attempt), 90.0))
            else:
                return None, False   # non-retriable (4xx other than 429)

        except Exception:
            return None, False       # network error, timeout, etc.

    return None, False


# ── Record-level labeling ─────────────────────────────────────────────────────

def label_record(
    client: anthropic.Anthropic,
    record: dict,
    id_counter: int,
    rate_limiter: _RateLimiter,
) -> tuple[Optional[SentimentRecord], bool, Optional[str]]:
    """
    Label one record. Returns (SentimentRecord, is_clean, error_msg).
    Never raises; errors are returned as the third tuple element.
    """
    text = record.get("text", "").strip()
    if not text:
        return None, False, "empty text"

    label_str, is_clean = call_api(client, text, rate_limiter)
    if label_str is None:
        return None, False, f"failed after retries: {text[:60]}"

    rec_id = record.get("id") or f"sfs-{id_counter:06d}"
    labeled = SentimentRecord(
        id=rec_id,
        text=text,
        label=Label(label_str),
        source=record.get("source", "unknown"),
        url=record.get("url", ""),
        date=record.get("date", ""),
        lang=record.get("lang", "sr"),
        label_source=LabelSource.llm_silver,
    )
    return labeled, is_clean, None


# ── Post-run reporting ────────────────────────────────────────────────────────

def print_distribution_and_examples(
    output_path: Path,
    seed: int = 42,
    n_examples: int = 8,
) -> None:
    """Print class distribution + n_examples random examples per class (4-class aware)."""
    import random
    rng = random.Random(seed)
    # Fixed order: financial labels first, filter label last
    label_order = ["positive", "negative", "neutral", "irrelevant"]
    by_label: dict[str, list[dict]] = {k: [] for k in label_order}

    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                lbl = rec.get("label", "")
                if lbl in by_label:
                    by_label[lbl].append(rec)
            except json.JSONDecodeError:
                pass

    total = sum(len(v) for v in by_label.values())
    if total == 0:
        return

    financial = sum(len(by_label[l]) for l in ["positive", "negative", "neutral"])
    n_irr = len(by_label["irrelevant"])

    print("\nClass distribution:")
    for lbl in label_order:
        recs = by_label[lbl]
        n = len(recs)
        if n == 0:
            continue
        pct = n / total * 100
        bar = "#" * int(pct / 2)
        tag = "  <- FILTER" if lbl == "irrelevant" else ""
        print(f"  {lbl:12s}: {n:5d}  ({pct:.1f}%)  {bar}{tag}")

    print(f"\n  Financial (pos+neg+neu) : {financial:5d}  ({financial/total:.1%})")
    print(f"  Irrelevant (filter)     : {n_irr:5d}  ({n_irr/total:.1%})")

    print(f"\n{n_examples} random examples per class (seed={seed}):")
    print("Verify: positive/negative/neutral = genuine market signal; irrelevant = no signal.")
    for lbl in label_order:
        recs = by_label[lbl]
        if not recs:
            continue
        shuffled = list(recs)
        rng.shuffle(shuffled)
        print(f"\n  [{lbl.upper()}]")
        for r in shuffled[:n_examples]:
            print(f"    {r['text'][:90]}")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_raw(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  Skipping line {lineno}: JSON error -- {exc}", file=sys.stderr)
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Silver-label JSONL with Claude Haiku")
    parser.add_argument("--input",   required=True, type=Path, help="Clean JSONL input")
    parser.add_argument("--output",  required=True, type=Path, help="Silver-labeled output")
    parser.add_argument(
        "--workers", "--max-workers",
        dest="workers",
        type=int, default=2,
        help="Parallel API threads (default: 2; reduce if still hitting 429s)",
    )
    parser.add_argument(
        "--rpm",
        type=int, default=45,
        help="Target requests per minute, shared across all workers "
             "(default: 45, safely under Tier 1 limit of 50/min)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records already present in --output (safe to re-run after interruption)",
    )
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example -> .env and fill it in.")

    client = anthropic.Anthropic(api_key=api_key)
    rate_limiter = _RateLimiter(rpm=args.rpm)

    raw_records = load_raw(args.input)
    print(f"Loaded {len(raw_records)} raw records from {args.input}")
    print(f"Workers: {args.workers}, RPM cap: {args.rpm}")

    # Resume: skip IDs already written
    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        print(f"Resuming -- {len(done_ids)} already done, skipping")

    to_label = [(i, r) for i, r in enumerate(raw_records) if r.get("id") not in done_ids]
    print(f"Records to label: {len(to_label)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    total_malformed = 0
    total_errors = 0
    total_success = 0

    with args.output.open(mode, encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(label_record, client, rec, idx, rate_limiter): (idx, rec)
                for idx, rec in to_label
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
                try:
                    labeled, is_clean, error = future.result()
                except Exception as exc:
                    # Should not reach here since label_record never raises,
                    # but guard anyway so a bug doesn't abort the whole run.
                    total_errors += 1
                    print(f"\n  Unexpected exception: {exc}", file=sys.stderr)
                    continue

                if error:
                    total_errors += 1
                    print(f"\n  Error: {error}", file=sys.stderr)
                    continue

                if not is_clean:
                    total_malformed += 1
                total_success += 1
                out_fh.write(labeled.to_jsonl_line() + "\n")
                out_fh.flush()

    print(f"\nDone.")
    print(f"  Successfully labeled : {total_success}")
    print(f"  Malformed responses  : {total_malformed} ({total_malformed/max(total_success,1):.1%})")
    print(f"  Errors / skipped     : {total_errors}")
    print(f"  Output               : {args.output}")

    print_distribution_and_examples(args.output)

    print()
    print("Next steps:")
    print("  1. python -m src.label.sample_gold   # draw balanced gold test sample")
    print("  2. Hand-label data/processed/test_to_label.jsonl (see data/ANNOTATION.md)")
    print("  3. python -m src.data.split --train-val-only")
    print("REMINDER: never use silver labels as test ground truth (SPEC.md §4.4).")


if __name__ == "__main__":
    main()
