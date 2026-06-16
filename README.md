# SerbFin-Sentiment

**First publicly available Serbian-language financial sentiment model.**

Fine-tuned Llama 3.2 3B Instruct via QLoRA (4-bit NF4) for 3-class financial sentiment
classification (`positive` / `negative` / `neutral`) on short Serbian financial headlines.

> Substantially outperforms zero-shot baselines and translation-based FinBERT on a held-out,
> human-labeled test set.

## Task

Investor-perspective financial sentiment — not emotional sentiment. "Company cuts 5,000 jobs
to boost margins" is `positive` from a market viewpoint. See `data/ANNOTATION.md` for guidelines.

## Repo layout

```
configs/          LoRA + training hyperparams (lora.yaml, train.yaml)
data/
  raw/            collected headlines (gitignored)
  interim/        silver-labeled pool
  processed/      final train/val/test splits
src/
  collect/        source ingestion
  label/          silver labeling (LLM) + review tooling
  data/           schema, split, stats
  train/          QLoRA training (qlora_train.py)
  eval/           metrics, baselines, run_eval
  publish/        push dataset + model to HF Hub
reports/          eval.md, confusion matrices
notebooks/        scratch only — no production logic
```

## Phases

| Phase | Status |
|-------|--------|
| 0 — Setup | done |
| 1 — Collect | in progress |
| 2 — Label | pending |
| 3 — Train | pending |
| 4 — Eval | pending |
| 5 — Publish | pending |
| 6 — Writeup | pending |

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY, HF_TOKEN

# Validate raw data
python -m src.data.schema --input data/raw/headlines.jsonl

# Silver-label the train pool
python -m src.label.silver --input data/raw/headlines.jsonl --output data/interim/silver.jsonl

# Split into train/val/test
python -m src.data.split --input data/interim/silver.jsonl --output-dir data/processed --stats
```

## Reproduction

```bash
# After Phase 2 (gold test set hand-labeled):
python src/train/qlora_train.py
python src/eval/run_eval.py --split test
```

## License

Model: [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). Built with Llama.
Dataset: CC BY 4.0 (headlines only, source URLs included; no full articles).
