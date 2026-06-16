"""
Draw a balanced stratified sample for the human gold test set.

From data/interim/silver.jsonl:
  - Sample min(target_per_class, minority_class_size) records per class.
    Default: 100 per class (~300 total). Reports if a class is short.
  - Write data/processed/test_to_label.jsonl:
      * Contains: id, text, source, url, date, lang, silver_hint
      * silver_hint = the silver-predicted label (for reference only)
      * NO 'label' field -- you fill that in from scratch after reading
        each headline carefully (investor/market perspective, see data/ANNOTATION.md)
      * NO 'label_source' field -- will be "human" after you annotate
  - Write data/interim/train_pool.jsonl:
      * All remaining records (not sampled), full SentimentRecord schema
      * label_source = "llm-silver"

CRITICAL (SPEC.md §4.4):
  The test set MUST be human-labeled. Never evaluate the final model on
  silver labels -- the comparison with zero-shot LLM would be circular.

Usage:
    python -m src.label.sample_gold
    python -m src.label.sample_gold --target-per-class 100 --seed 42
    python -m src.label.sample_gold --silver data/interim/silver.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


TARGET_PER_CLASS = 100
SEED = 42
LABELS = ("positive", "negative", "neutral")
# Records with these labels are filter-only; they never go into test or train pool.
FILTER_LABELS = {"irrelevant"}


def load_silver(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: {exc}", file=sys.stderr)
    return records


def sample_gold(
    silver_path: Path,
    test_path: Path,
    train_pool_path: Path,
    target_per_class: int = TARGET_PER_CLASS,
    seed: int = SEED,
) -> None:
    records = load_silver(silver_path)
    print(f"Loaded {len(records)} silver records from {silver_path}")

    by_label: dict[str, list[dict]] = defaultdict(list)
    skipped_filter = 0
    unlabeled = []
    for rec in records:
        lbl = rec.get("label", "")
        if lbl in LABELS:
            by_label[lbl].append(rec)
        elif lbl in FILTER_LABELS:
            skipped_filter += 1  # silently drop filter labels from both pools
        else:
            unlabeled.append(rec)

    if skipped_filter:
        print(f"  Excluded {skipped_filter} filter-label records (irrelevant) from both pools.")
    if unlabeled:
        print(f"  [warn] {len(unlabeled)} records with unrecognized label -- moved to train pool")

    # Report class sizes
    print("\nFinancial silver label distribution (pos/neg/neu only):")
    for lbl in LABELS:
        n = len(by_label[lbl])
        print(f"  {lbl:10s}: {n:5d}")

    min_n = min(len(by_label[lbl]) for lbl in LABELS)
    actual_per_class = min(target_per_class, min_n)

    if actual_per_class < target_per_class:
        min_lbl = min(LABELS, key=lambda l: len(by_label[l]))
        print(
            f"\n  WARNING: minority class '{min_lbl}' has only {min_n} records. "
            f"Capping sample at {actual_per_class} per class."
        )
    else:
        print(f"\n  Sampling {actual_per_class} per class (target: {target_per_class}).")

    rng = random.Random(seed)
    sampled_ids: set[str] = set()
    test_records: list[dict] = []

    for lbl in LABELS:
        pool = list(by_label[lbl])
        rng.shuffle(pool)
        for rec in pool[:actual_per_class]:
            sampled_ids.add(rec["id"])
            # Strip label/label_source; add silver_hint so the annotator
            # can cross-check without being anchored to it.
            test_records.append({
                "id":          rec["id"],
                "text":        rec["text"],
                "source":      rec.get("source", ""),
                "url":         rec.get("url", ""),
                "date":        rec.get("date", ""),
                "lang":        rec.get("lang", "sr"),
                "silver_hint": lbl,
                # "label":  <-- intentionally absent; you fill this in
                # "label_source": <-- will be "human" after annotation
            })

    rng.shuffle(test_records)  # don't leak class order to the annotator

    # Financial records not sampled into test set go to train pool.
    # Filter-label records (irrelevant) are excluded entirely.
    train_pool = [
        r for r in records
        if r.get("id") not in sampled_ids and r.get("label") not in FILTER_LABELS
    ]
    train_pool.extend(unlabeled)

    # Write outputs
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_pool_path.parent.mkdir(parents=True, exist_ok=True)

    with test_path.open("w", encoding="utf-8") as fh:
        for rec in test_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with train_pool_path.open("w", encoding="utf-8") as fh:
        for rec in train_pool:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nGold test sample written: {test_path}")
    print(f"  Total records      : {len(test_records)}")
    for lbl in LABELS:
        n = sum(1 for r in test_records if r["silver_hint"] == lbl)
        print(f"  {lbl:10s}         : {n}")

    print(f"\nTrain pool written: {train_pool_path}")
    print(f"  Total records      : {len(train_pool)}")

    print()
    print("Next steps:")
    print("  1. Open data/processed/test_to_label.jsonl")
    print("  2. For each record, read the headline and add:")
    print('     "label": "positive"  OR  "negative"  OR  "neutral"')
    print('     "label_source": "human"')
    print("  3. Judge from INVESTOR/MARKET perspective (not emotional).")
    print("     See data/ANNOTATION.md for guidelines and common traps.")
    print("  4. Aim for a second annotator on at least 50 items -> compute Cohen's kappa.")
    print("  5. When done: python -m src.data.split --train-val-only")
    print()
    print("  silver_hint is provided for reference. Trust your own judgment over it.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Sample balanced gold test set from silver-labeled pool"
    )
    parser.add_argument(
        "--silver",
        type=Path,
        default=Path("data/interim/silver.jsonl"),
        help="Silver-labeled input (default: data/interim/silver.jsonl)",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=Path("data/processed/test_to_label.jsonl"),
        help="Output for human annotation (default: data/processed/test_to_label.jsonl)",
    )
    parser.add_argument(
        "--train-pool-out",
        type=Path,
        default=Path("data/interim/train_pool.jsonl"),
        help="Remaining records for training (default: data/interim/train_pool.jsonl)",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=TARGET_PER_CLASS,
        help=f"Target samples per class (default: {TARGET_PER_CLASS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    args = parser.parse_args()

    if not args.silver.exists():
        sys.exit(f"Silver file not found: {args.silver}\nRun silver labeling first.")

    sample_gold(
        silver_path=args.silver,
        test_path=args.test_out,
        train_pool_path=args.train_pool_out,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
