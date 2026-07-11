# SerbFin-Sentiment

**First publicly available Serbian-language financial sentiment model.**

Fine-tuned Llama 3.2 3B Instruct via QLoRA (4-bit NF4) for 3-class financial sentiment
classification (`positive` / `negative` / `neutral`) on short Serbian financial headlines.

> Substantially outperforms zero-shot baselines and translation-based FinBERT on a held-out,
> human-labeled test set.

**[Model on HF Hub](https://huggingface.co/filipovicfilip222/serbfin-sentiment-llama-3.2-3b)** ·
**[Dataset on HF Hub](https://huggingface.co/datasets/filipovicfilip222/serbfin-sentiment)**

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
| 1 — Collect | done |
| 2 — Label | done (240 human-labeled gold test set + silver train pool) |
| 3 — Train | done — QLoRA adapter trained on RunPod (RTX A5000), 3 epochs, ~84s |
| 4 — Eval | **done — success criterion met** (fine-tuned beats both baselines) |
| 5 — Publish | done — [dataset](https://huggingface.co/datasets/filipovicfilip222/serbfin-sentiment) + [model](https://huggingface.co/filipovicfilip222/serbfin-sentiment-llama-3.2-3b) on HF Hub |
| 6 — Writeup | pending |

### Results (Phase 4) — SUCCESS

All three scored on the same 240-headline human-labeled gold test set:

| Model | macro-F1 | accuracy | malformed rate |
|---|---|---|---|
| Zero-shot Llama 3.2 3B | 0.6389 | 0.6417 | 0.00% |
| Translate (sr→en) + FinBERT | 0.5434 | 0.5417 | 0.00% |
| **Fine-tuned QLoRA adapter** | **0.7379** | 0.7333 | 0.00% |

Fine-tuned macro-F1 (0.7379) beats both required baselines (SPEC.md §8 success criterion). Full
per-class breakdown and confusion matrices in `reports/eval.md`. Adapter weights published at
[filipovicfilip222/serbfin-sentiment-llama-3.2-3b](https://huggingface.co/filipovicfilip222/serbfin-sentiment-llama-3.2-3b)
(local copy in `outputs/serbfin-llama-3.2-3b/final_adapter/` is gitignored, GitHub only carries code).

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
# Baselines, scored against the same gold data/processed/test.jsonl
# (needs a GPU + HF_TOKEN with access to meta-llama/Llama-3.2-3B-Instruct for the first one):
python -m src.eval.zero_shot_llama
python -m src.eval.finbert_baseline

# Fine-tuning (needs a CUDA GPU -- bitsandbytes 4-bit has no CPU/AMD path):
python -m src.data.split --train-val-only --stats
python -m src.train.qlora_train

# Eval the fine-tuned adapter against the same gold set + baselines:
python -m src.eval.finetuned_eval
```

All eval scripts write to `reports/predictions_*.jsonl` and merge their scores into
`reports/eval.md` (macro-F1, per-class precision/recall/F1, confusion matrix, malformed-output
rate). Re-running any one script only replaces its own section in the report — the other two
are left untouched. (Caution: if you copy `reports/` wholesale from a different machine, e.g.
a rented GPU pod that only ran one script, you'll overwrite the merged file with a partial one.
Prefer copying just the specific `predictions_*.jsonl` you produced there and re-merge locally.)

## Publish

```bash
# Requires HF_TOKEN in .env with WRITE access (a "Read"-only token will 403):
python -m src.publish.push_dataset
python -m src.publish.push_model
```

Both are CPU-only (no GPU needed) -- they read local files and upload. `push_model` writes a
generated model card straight into `outputs/serbfin-llama-3.2-3b/final_adapter/README.md` before
uploading, pulling the eval summary table live from `reports/eval.md`.

## License

Model: [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). Built with Llama.
Dataset: CC BY 4.0 (headlines only, source URLs included; no full articles).
