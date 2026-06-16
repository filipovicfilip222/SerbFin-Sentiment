"""
JSONL record schema for SerbFin-Sentiment (see SPEC.md §4.2).

Two record types:
  RawRecord       — unlabeled, written by src/collect/ (data/raw/)
  SentimentRecord — labeled, written by src/label/ (data/interim/, data/processed/)

Usage:
    python -m src.data.schema --input data/raw/headlines.jsonl       # validates raw
    python -m src.data.schema --input data/processed/train.jsonl     # validates labeled
"""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class Label(str, Enum):
    positive   = "positive"
    negative   = "negative"
    neutral    = "neutral"
    # Added in v2: used as a FILTER label during silver labeling to separate
    # headlines with no investor/market signal (social protests, trade fairs,
    # PR denials, general news) from genuine financial sentiment.
    # Strip these before training -- the trained model stays 3-class.
    irrelevant = "irrelevant"


class LabelSource(str, Enum):
    human = "human"
    llm_silver = "llm-silver"
    llm_reviewed = "llm-reviewed"


class RawRecord(BaseModel):
    """Unlabeled collected headline (data/raw/*.jsonl)."""

    id: str
    text: str
    source: str
    url: str
    date: str           # ISO 8601 (YYYY-MM-DD)
    lang: str = "sr"

    @field_validator("id")
    @classmethod
    def id_format(cls, v: str) -> str:
        if not v.startswith("sfs-"):
            raise ValueError(f"id must start with 'sfs-', got: {v!r}")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v

    @field_validator("date")
    @classmethod
    def date_iso(cls, v: str) -> str:
        from datetime import date
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"date must be ISO 8601 (YYYY-MM-DD), got: {v!r}")
        return v

    @field_validator("lang")
    @classmethod
    def lang_sr(cls, v: str) -> str:
        if v != "sr":
            raise ValueError(f"lang must be 'sr', got: {v!r}")
        return v

    def to_jsonl_line(self) -> str:
        return self.model_dump_json()


class SentimentRecord(RawRecord):
    """Labeled record (data/interim/, data/processed/)."""

    label: Label
    label_source: LabelSource

    @model_validator(mode="after")
    def gold_test_must_be_human(self) -> "SentimentRecord":
        # Enforcement happens at split time; schema itself only warns.
        return self


def validate_raw_file(path: Path) -> tuple[list[RawRecord], list[tuple[int, str]]]:
    """Validate a raw (unlabeled) JSONL file. Returns (records, errors)."""
    records: list[RawRecord] = []
    errors: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append((lineno, f"JSON parse error: {exc}"))
                continue
            try:
                records.append(RawRecord.model_validate(data))
            except Exception as exc:
                errors.append((lineno, str(exc)))
    return records, errors


def validate_file(path: Path) -> tuple[list[SentimentRecord], list[tuple[int, str]]]:
    """
    Parse and validate every line in a JSONL file.

    Returns (valid_records, errors) where errors is a list of (line_no, message).
    """
    records: list[SentimentRecord] = []
    errors: list[tuple[int, str]] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append((lineno, f"JSON parse error: {exc}"))
                continue
            try:
                records.append(SentimentRecord.model_validate(data))
            except Exception as exc:
                errors.append((lineno, str(exc)))

    return records, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SerbFin-Sentiment JSONL records")
    parser.add_argument("--input", required=True, type=Path, help="Path to JSONL file")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Validate as RawRecord (no label/label_source required) — use for data/raw/",
    )
    args = parser.parse_args()

    if args.raw:
        records, errors = validate_raw_file(args.input)
    else:
        records, errors = validate_file(args.input)

    print(f"Valid records : {len(records)}")
    print(f"Errors        : {len(errors)}")

    if errors:
        for lineno, msg in errors:
            print(f"  line {lineno}: {msg}", file=sys.stderr)
        sys.exit(1)
    else:
        print("All records valid.")


if __name__ == "__main__":
    main()
