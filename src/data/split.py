"""
Stratified train/val split of the silver-labeled train pool (seed=42).

The test set is NOT produced here -- it comes from the human-gold-labeled
data/processed/test_to_label.jsonl (after you annotate it).
See SPEC.md §4.3 and the gold-test-set rule in §4.4.

Default mode (--train-val-only, for use after sample_gold.py):
  Input : data/interim/train_pool.jsonl   (silver labels, never test items)
  Output: data/processed/train.jsonl  (70 %)
          data/processed/val.jsonl    (30 %)

Legacy mode (no flag, produces a silver test split -- only for debugging):
  Output also includes data/processed/test_silver.jsonl  (15 % silver labels)
  WARNING: do NOT use test_silver.jsonl for final evaluation.

Usage:
    # Standard -- use after sample_gold.py
    python -m src.data.split --train-val-only
    python -m src.data.split --train-val-only --stats

    # With explicit paths
    python -m src.data.split --train-val-only \\
        --input data/interim/train_pool.jsonl --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from src.data.schema import Label, SentimentRecord, validate_file


SEED = 42


def stratified_split_2way(
    records: list[SentimentRecord],
    train_ratio: float = 0.70,
    seed: int = SEED,
) -> tuple[list[SentimentRecord], list[SentimentRecord]]:
    """Return (train, val) with stratification by label, seed=42."""
    rng = random.Random(seed)
    by_label: dict[str, list[SentimentRecord]] = defaultdict(list)
    for rec in records:
        by_label[rec.label.value].append(rec)

    train, val = [], []
    for group in by_label.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_train = round(len(shuffled) * train_ratio)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def stratified_split_3way(
    records: list[SentimentRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[list[SentimentRecord], list[SentimentRecord], list[SentimentRecord]]:
    """Return (train, val, test) with stratification by label."""
    rng = random.Random(seed)
    by_label: dict[str, list[SentimentRecord]] = defaultdict(list)
    for rec in records:
        by_label[rec.label.value].append(rec)

    train, val, test = [], [], []
    for group in by_label.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(records: list[SentimentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_jsonl_line() + "\n")


def print_split_stats(splits: list[tuple[str, list[SentimentRecord]]]) -> None:
    total = sum(len(recs) for _, recs in splits)
    print(f"\nTotal : {total}")
    for name, recs in splits:
        n = len(recs)
        counts: dict[str, int] = defaultdict(int)
        for rec in recs:
            counts[rec.label.value] += 1
        print(f"\n  {name} ({n}, {n/total:.1%}):")
        for lbl in Label:
            c = counts.get(lbl.value, 0)
            bar = "#" * int(c / max(n, 1) * 30)
            print(f"    {lbl.value:10s}: {c:4d}  ({c/max(n,1):.1%})  {bar}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Stratified train/val split for SerbFin")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/train_pool.jsonl"),
        help="Labeled JSONL to split (default: data/interim/train_pool.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--train-val-only",
        action="store_true",
        help=(
            "70/30 train/val split only -- no test split. "
            "Use this: the test set comes from your human-labeled "
            "data/processed/test_to_label.jsonl, NOT from here."
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-class counts and label distribution for each split",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    records, errors = validate_file(args.input)
    if errors:
        for lineno, msg in errors:
            print(f"  line {lineno}: {msg}")
        sys.exit(f"{len(errors)} schema errors in {args.input}. Fix before splitting.")

    if args.train_val_only:
        train, val = stratified_split_2way(records)

        write_jsonl(train, args.output_dir / "train.jsonl")
        write_jsonl(val,   args.output_dir / "val.jsonl")

        print(f"Splits written to {args.output_dir}/")
        print(f"  train.jsonl : {len(train)} records  (70 %)")
        print(f"  val.jsonl   : {len(val)} records  (30 %)")
        print()
        print("Test set: use your human-annotated data/processed/test_to_label.jsonl")
        print("  Add 'label' and 'label_source: human' to each row, then rename to test.jsonl.")

        if args.stats:
            print_split_stats([("train", train), ("val", val)])

    else:
        # Legacy 70/15/15 -- only for debugging, never for final eval
        train, val, test = stratified_split_3way(records)

        write_jsonl(train, args.output_dir / "train.jsonl")
        write_jsonl(val,   args.output_dir / "val.jsonl")
        write_jsonl(test,  args.output_dir / "test_silver.jsonl")

        print(f"Splits written to {args.output_dir}/")
        print(f"  train.jsonl        : {len(train)}")
        print(f"  val.jsonl          : {len(val)}")
        print(f"  test_silver.jsonl  : {len(test)}  <-- silver labels, NOT for final eval")
        print()
        print("WARNING: test_silver.jsonl uses LLM-generated labels.")
        print("  For the publishable result, use the human-gold test set.")

        if args.stats:
            print_split_stats([("train", train), ("val", val), ("test_silver", test)])


if __name__ == "__main__":
    main()
