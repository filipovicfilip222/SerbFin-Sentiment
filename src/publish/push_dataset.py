"""
Push the SerbFin-Sentiment dataset to the HF Hub (SPEC.md §7.1, §7.3).

Publishes data/processed/{train,val,test}.jsonl as a 3-split dataset repo
(train / validation / test), plus a dataset card (README.md) with sources,
labeling method, schema, split sizes, class balance, and licensing -- all
computed FROM the actual data files, not hand-typed, so the card can't drift
out of sync with reality.

Requires HF_TOKEN in .env with WRITE access (a "Read"-only token will fail
with a 403). Create one at https://huggingface.co/settings/tokens.

Usage:
    python -m src.publish.push_dataset
    python -m src.publish.push_dataset --repo-id filipovicfilip222/serbfin-sentiment
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_REPO_ID = "filipovicfilip222/serbfin-sentiment"
LABELS = ("positive", "negative", "neutral")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def class_balance(records: list[dict]) -> dict:
    c = Counter(r["label"] for r in records)
    return {lab: c.get(lab, 0) for lab in LABELS}


def date_range(records: list[dict]) -> tuple[str, str]:
    dates = sorted(r["date"] for r in records if r.get("date"))
    return (dates[0], dates[-1]) if dates else ("unknown", "unknown")


def build_card(train: list[dict], val: list[dict], test: list[dict], repo_id: str) -> str:
    all_records = train + val + test
    sources = sorted({r.get("source", "unknown") for r in all_records})
    lo, hi = date_range(all_records)

    def balance_table(recs: list[dict]) -> str:
        n = len(recs)
        c = class_balance(recs)
        lines = ["| label | count | % |", "|---|---|---|"]
        for lab in LABELS:
            pct = c[lab] / n * 100 if n else 0
            lines.append(f"| {lab} | {c[lab]} | {pct:.1f}% |")
        return "\n".join(lines)

    return f"""---
language:
- sr
license: cc-by-4.0
task_categories:
- text-classification
tags:
- financial-sentiment
- serbian
- sentiment-analysis
pretty_name: SerbFin-Sentiment
---

# SerbFin-Sentiment

First publicly available **Serbian-language financial sentiment** dataset. Short financial
news headlines / lead sentences from Serbian business news sites, labeled for 3-class sentiment
from an **investor / market perspective** (not emotional sentiment).

> "Company cuts 5,000 jobs to boost margins" is `positive` from a market viewpoint even though
> it sounds bad. See the annotation guidelines below.

## Sources & collection method

Collected from Serbian financial news sites: {", ".join(sources)}.
Date range: {lo} to {hi}.

Only short headline/lead text, source name, source URL, and date are stored -- **no full
article text is redistributed**, per each site's terms of use.

## Labeling

Three-stage weak-supervision -> human-gold pipeline (SPEC.md §4.4):

1. **Silver labels** -- Claude Haiku pre-labels the train pool (`label_source: "llm-silver"`).
2. **Gold test set** -- 240 headlines, stratified sample, hand-labeled from scratch by a human
   annotator (`label_source: "human"`), **never** auto-labeled. The silver hint was hidden from
   the annotator until after each label was chosen (see `data/ANNOTATION.md` in the GitHub repo
   for full guidelines and the labeling tool used).

> Test labels are 100% human -- this is what makes the "beats zero-shot LLM" comparison in the
> model card valid rather than circular (evaluating an LLM against its own silver labels would
> just measure self-agreement, not accuracy).

## Label schema

| label | meaning (investor / market perspective) |
|---|---|
| `positive` | Bullish -- good for company/sector/economy: profit up, beat expectations, new deal, growth, upgrade |
| `negative` | Bearish -- bad for company/sector/economy: loss, miss, layoffs, downgrade, default, legal penalty |
| `neutral` | No clear directional signal: appointment, schedule, mixed result, purely descriptive |

## Splits

| split | n | source |
|---|---|---|
| train | {len(train)} | llm-silver |
| validation | {len(val)} | llm-silver |
| test | {len(test)} | **human** (gold) |

### Class balance

**train**
{balance_table(train)}

**validation**
{balance_table(val)}

**test (gold)**
{balance_table(test)}

Financial news skews `neutral`; no oversampling was applied to val/test (only ever appropriate
for train, and not used here).

## Inter-annotator agreement

Not computed for this release -- the gold test set was single-annotated. A second annotation
pass (Cohen's kappa >= 0.6 target per SPEC.md §4.3) is left as future work.

## License

CC BY 4.0. Headlines only, with source URLs included; no full articles are redistributed.

## Intended use & limitations

Intended for training/evaluating Serbian financial sentiment models. Domain is short financial
news headlines, not social media or emotional text -- performance on out-of-domain text
(tweets, long-form articles, non-financial news) is untested. Class balance skews neutral;
the `positive`/`neutral` boundary is the most common source of model confusion (see the
companion model's confusion matrix at
[{repo_id.rsplit("/", 1)[0]}/serbfin-sentiment-llama-3.2-3b](https://huggingface.co/{repo_id.rsplit("/", 1)[0]}/serbfin-sentiment-llama-3.2-3b)).

## Companion model

[{repo_id.rsplit("/", 1)[0]}/serbfin-sentiment-llama-3.2-3b](https://huggingface.co/{repo_id.rsplit("/", 1)[0]}/serbfin-sentiment-llama-3.2-3b) --
QLoRA fine-tuned Llama 3.2 3B Instruct adapter trained on this dataset's `train`/`validation`
splits and evaluated on `test`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Push SerbFin-Sentiment dataset to HF Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--private", action="store_true", help="Push as a private repo")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN not set in .env")

    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, create_repo

    train = load_jsonl(args.train)
    val = load_jsonl(args.val)
    test = load_jsonl(args.test)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    # Gold test set must be human-labeled -- refuse to publish otherwise (SPEC.md §4.4).
    non_human = [r for r in test if r.get("label_source") != "human"]
    if non_human:
        raise SystemExit(
            f"{len(non_human)} test records are not label_source='human'. "
            "Refusing to publish a non-gold test set."
        )

    dd = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        }
    )

    print(f"Creating/using repo {args.repo_id} (dataset)...")
    create_repo(args.repo_id, token=hf_token, repo_type="dataset", exist_ok=True, private=args.private)

    print("Pushing splits...")
    dd.push_to_hub(args.repo_id, token=hf_token, private=args.private)

    print("Pushing dataset card (README.md)...")
    card = build_card(train, val, test, args.repo_id)
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
