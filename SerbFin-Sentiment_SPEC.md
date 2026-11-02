# SerbFin-Sentiment — Project Specification

> **Note:** this is the original design spec, written before the work started, and is kept as a
> record of the plan. Some of its goals were not met or not verified (for example double
> annotation with Cohen's κ ≥ 0.6, and pinned dependencies). See [README.md](README.md) for what
> was actually done and its limitations.

> Fine-tuned Serbian financial sentiment classification model.
> Base: **Llama 3.2 3B Instruct** · Method: **QLoRA (4-bit NF4)** · Stack: **PyTorch · PEFT · TRL · HuggingFace**

This is the single source of truth for the project. It is written for a handoff to Claude Code / Cursor and for the academic/portfolio writeup. Treat decisions here as authoritative; if you deviate, update this file.

---

## 1. Goal & positioning

Build a **Serbian-language financial sentiment model**, published on the HuggingFace Hub together with its labeled dataset.

The headline claim for the CV / model card is:

> "Substantially outperforms zero-shot baselines and translation-based FinBERT on a held-out, human-labeled test set."

That claim is only defensible if the **test set is human gold-labeled** (see §4). Everything in this spec is designed to make that claim true and reproducible.

### Task definition
Three-class **financial sentiment** classification on short Serbian financial texts (headlines / lead sentences):

| Label | Meaning (investor / market perspective) |
|-------|------------------------------------------|
| `positive` | Bullish — good for the company / sector / economy (profit up, beat expectations, growth, new deal) |
| `negative` | Bearish — bad for the company / sector / economy (loss, miss, layoffs, downgrade, default) |
| `neutral` | Factual / no clear directional signal (appointment, schedule, mixed, purely descriptive) |

> **Critical nuance:** this is *financial* sentiment, not *emotional* sentiment. "Company cuts 5,000 jobs to boost margins" is often `positive` from a market viewpoint even though it sounds bad. Annotators must judge from the investor's perspective, in the style of the Financial PhraseBank. This single rule is the most common source of label noise.

---

## 2. Deliverables

1. **GitHub repo** — training + eval + publishing code (this spec lives in it as `SPEC.md`, plus a `CLAUDE.md`).
2. **HF dataset repo** — `<username>/serbfin-sentiment` with `dataset_card.md`.
3. **HF model repo** — `<username>/serbfin-sentiment-llama-3.2-3b` (LoRA adapter + optional merged weights) with `model_card.md`.
4. **Eval report** — `reports/eval.md`: metrics table, confusion matrices, baseline comparison.
5. **One-paragraph CV bullet + a short blog/writeup** (reuse for the academic commission).

---

## 3. Repository layout

```
serbfin-sentiment/
├── SPEC.md                  # this file — source of truth
├── CLAUDE.md                # condensed context for Claude Code (see §11)
├── README.md
├── pyproject.toml           # or requirements.txt
├── .env.example             # HF_TOKEN, ANTHROPIC_API_KEY / AWS creds
├── configs/
│   ├── lora.yaml            # LoRA + quant hyperparams
│   └── train.yaml           # SFTConfig hyperparams
├── data/
│   ├── raw/                 # collected items (jsonl), gitignored
│   ├── interim/             # weak-labeled (silver)
│   └── processed/           # final splits: train.jsonl, val.jsonl, test.jsonl
├── src/
│   ├── collect/             # source ingestion (headlines + metadata)
│   ├── label/               # weak labeling (LLM) + human-review tooling
│   ├── data/                # schema, split, stats, balance
│   ├── train/               # qlora_train.py
│   ├── eval/                # metrics.py, baselines.py, run_eval.py
│   └── publish/             # push_dataset.py, push_model.py
├── notebooks/               # exploration only — no production logic here
└── reports/
    └── eval.md
```

**Rule:** production logic lives in `src/`, never in notebooks. Notebooks are scratch.

---

## 4. Data

### 4.1 Sources
Serbian financial news headlines / lead sentences. Candidate sources:
`Kamatica`, `Biznis.rs`, `eKapija`, `Blic Biznis`, `B92 Biznis`, `Politika — Ekonomija`, `NBS` announcements, `BELEX` market news.

> **Legal / copyright:** Do **not** redistribute full articles. The dataset stores: the **short headline text** (a few words — fair-use-ish, but keep it minimal), the **label**, the **source name**, the **source URL**, and the **date**. When in doubt, store a lightly **paraphrased** headline + link to original. State the collection method and date range in the dataset card. Respect each site's `robots.txt` and ToS.

### 4.2 Schema (JSONL, one object per line)
```json
{
  "id": "sfs-000123",
  "text": "NIS objavio rekordnu dobit u prvom kvartalu",
  "label": "positive",
  "source": "eKapija",
  "url": "https://...",
  "date": "2025-04-12",
  "lang": "sr",
  "label_source": "human"   // "human" | "llm-silver" | "llm-reviewed"
}
```

### 4.3 Size & splits
- **v1 target:** 1,500–3,000 labeled items. Quality over quantity — 1,500 clean beats 10,000 noisy.
- **Stratified split** by label: `train 70% / val 15% / test 15%`.
- **Test set must be 100% human gold-labeled.** Ideally double-annotated; report inter-annotator agreement (Cohen's κ). Target κ ≥ 0.6.
- Financial news skews `neutral` — track class balance and report it. Consider mild oversampling of minority classes in training only (never in val/test).

### 4.4 Labeling strategy (weak supervision → human gold)
You have Bedrock (Claude Haiku) and Anthropic API access — use them, but carefully.

1. **Silver labels:** Claude pre-labels the train pool (`label_source: "llm-silver"`). Cheap, fast, gets you 80% of the way.
2. **Human review of train:** spot-check and fix obvious silver errors → `"llm-reviewed"`.
3. **Gold test set:** **never** auto-labeled. Hand-label (you + ideally one more annotator). This is non-negotiable — see the warning below.

> ⚠️ **Methodological trap:** if you want to claim "beats zero-shot LLM," you **cannot** evaluate on LLM-generated labels. The test labels must be human. Otherwise you're measuring agreement-with-the-LLM, not accuracy, and the comparison is circular.

Annotation guidelines (put in `data/ANNOTATION.md`):
- Judge from investor/market perspective, not emotion.
- Mixed sentiment → dominant directional signal; if truly balanced → `neutral`.
- Pure facts, schedules, appointments → `neutral`.
- Hedged / speculative ("could", "may") → weight the realized framing, default `neutral` if no clear direction.

---

## 5. Modeling

### 5.1 Approach
**Instruction-tuned generative classification** (matches the "fine-tuned LLM" framing and is more publishable than a vanilla classification head). The model is trained to emit a single label token given an instruction + the text. Loss is computed on the **completion only**.

Prompt template (keep it fixed and identical at train + eval + for the zero-shot baseline):
```
<system>
Ti si klasifikator finansijskog sentimenta. Za dati naslov vesti odredi sentiment sa perspektive investitora.
Odgovori isključivo jednom rečju: positive, negative ili neutral.
</system>
<user>
Naslov: {text}
Sentiment:
</user>
<assistant>
{label}
```
(Render this via the tokenizer's chat template — do not hand-build the special tokens.)

> Alternative considered: `AutoModelForSequenceClassification` + LoRA (cleaner metrics, no parsing). Rejected as primary because the generative route is the stronger portfolio story and lets you reuse the exact same prompt for the zero-shot baseline. Keep it in your back pocket if generative output parsing gets noisy.

### 5.2 Quantization (QLoRA)
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### 5.3 LoRA config (`configs/lora.yaml`)
```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
bias: none
task_type: CAUSAL_LM
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### 5.4 Training config (`configs/train.yaml`)
```yaml
output_dir: outputs/serbfin-llama-3.2-3b
num_train_epochs: 3
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
optim: paged_adamw_8bit
bf16: true
gradient_checkpointing: true
max_length: 512        # headlines are short
logging_steps: 10
eval_strategy: steps
eval_steps: 50
save_strategy: steps
save_steps: 50
load_best_model_at_end: true
metric_for_best_model: eval_loss
seed: 42
completion_only_loss: true   # loss on the label only
report_to: none              # or "wandb"
```

> **Defaults that work:** r=16, alpha≈16–32, lr=2e-4, 2–3 epochs. Start here, then experiment. Watch for overfitting after epoch 2 on a small dataset — early stopping on `eval_loss` is your friend.

### 5.5 ⚠️ Current TRL API (important — most tutorials are stale)
As of TRL **≥ 0.16**:
- `TrainingArguments` → use **`SFTConfig`**.
- `tokenizer=` argument was **removed** → use **`processing_class=`**.
- Set `tokenizer.pad_token = tokenizer.eos_token` (Llama has no pad token) and align padding side.

Correct trainer init:
```python
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(**train_cfg),        # NOT TrainingArguments
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=LoraConfig(**lora_cfg),
    processing_class=tokenizer,         # NOT tokenizer=
)
trainer.train()
```

### 5.6 Hardware
3B in 4-bit fits comfortably in **~10–12 GB VRAM**. Options, cheapest first:
- **Kaggle / Colab T4 (free)** — fine for v1.
- **AWS** (you have access): a single `g5.xlarge` (A10G, 24 GB) or SageMaker training job. Overkill for v1 but good for the "deployed on AWS" story.

---

## 6. Evaluation

### 6.1 Metrics
- **Macro-F1** (headline metric — robust to the neutral skew).
- Accuracy.
- Per-class precision / recall / F1.
- Confusion matrix (save as `reports/confusion_*.png`).

### 6.2 Baselines (run on the *same* gold test set)
1. **Zero-shot Llama 3.2 3B Instruct** — identical prompt, no fine-tuning. This is the "did fine-tuning actually help" baseline.
2. **Translation → FinBERT** — translate `sr → en` (e.g., a NLLB / Opus-MT model or a translation API), then classify with `ProsusAI/finbert`. This is the "did you beat the obvious lazy pipeline" baseline.
3. *(optional)* Multilingual sentiment model (e.g. an XLM-R sentiment checkpoint) for a stronger reference.

### 6.3 Output parsing (generative → label)
The model emits free text. Two robust options:
- **Constrained / first-token scoring:** compare logits of the `positive`/`negative`/`neutral` first tokens — deterministic, no parsing.
- **Regex fallback:** lowercase, strip, match the first of the three label words; if none → count as wrong (don't silently drop).

Log the **fraction of malformed outputs** — it should be ~0 after fine-tuning, and is itself a result (fine-tuning makes the model format-obedient vs. the zero-shot baseline).

---

## 7. Publishing

### 7.1 Dataset card (`dataset_card.md`) must include
Sources & collection method, date range, label schema + the financial-perspective definition, split sizes, class balance, inter-annotator κ, licensing note, intended use & limitations.

### 7.2 Model card (`model_card.md`) must include
Base model + license note (Llama 3.2 Community License — comply with its naming/attribution terms), training data, hyperparameters, the eval table vs. all baselines, intended use, limitations & biases, how-to-load snippet, reproduction command.

### 7.3 Push
```python
# adapter
model.push_to_hub("<username>/serbfin-sentiment-llama-3.2-3b")
tokenizer.push_to_hub("<username>/serbfin-sentiment-llama-3.2-3b")
# dataset
dataset.push_to_hub("<username>/serbfin-sentiment")
```

> **Llama license:** Llama 3.2 derivatives must carry "Llama" in the name and include the license / "Built with Llama" attribution. Keep `llama` in the repo name and add the notice to the model card.

---

## 8. Phased plan

| Phase | Output | Done when |
|-------|--------|-----------|
| 0 — Setup | repo, env, `.env`, HF + API auth, deps pinned | `python -c "import torch, trl, peft, transformers"` works; HF login OK |
| 1 — Collect | `data/raw/*.jsonl`, ≥ 2,000 candidate items | dedup'd, schema-valid |
| 2 — Label | silver train + **gold test**, `ANNOTATION.md`, κ reported | splits in `data/processed/`, κ ≥ 0.6 |
| 3 — Train | trained adapter, training curves | eval_loss converges, no NaN, format obedience high |
| 4 — Eval | `reports/eval.md` with all 3 baselines + confusion matrices | fine-tuned macro-F1 > both required baselines |
| 5 — Publish | HF dataset + model repos, both cards | repos load end-to-end from a clean env |
| 6 — Writeup | CV bullet + short blog / commission slides | reproducible from README in one command |

---

## 9. Pitfalls checklist
- [ ] Test set is **human** labeled (not LLM).
- [ ] Same prompt template at train / zero-shot / eval.
- [ ] `pad_token` set; padding side correct for causal LM.
- [ ] Using `SFTConfig` + `processing_class` (not the deprecated API).
- [ ] Loss on completion only (`completion_only_loss=True`).
- [ ] Macro-F1 reported (not just accuracy — neutral skew inflates accuracy).
- [ ] Malformed-output rate logged.
- [ ] Seeds fixed; library versions pinned.
- [ ] Llama license attribution in model card + name.
- [ ] No full articles redistributed in the dataset.

---

## 10. Dependencies (pin after a clean install resolves them)
```
torch
transformers>=4.45
trl>=0.16          # SFTConfig + processing_class API
peft>=0.13
bitsandbytes>=0.44
accelerate>=1.0
datasets>=3.0
evaluate
scikit-learn
huggingface_hub
anthropic          # for silver labeling
```
> Don't blind-copy versions from old blog posts — they pin pre-0.12 TRL and will break on `processing_class`. Install latest compatible, then freeze with `pip freeze > requirements.lock`.

---

## 11. `CLAUDE.md` (drop this condensed file at repo root for Claude Code)

```markdown
# SerbFin-Sentiment — Agent Context

Serbian-language financial sentiment model (Llama 3.2 3B, QLoRA).
Base: Llama 3.2 3B Instruct · QLoRA 4-bit NF4 · PyTorch/PEFT/TRL/HF.
Full detail: see SPEC.md (source of truth).

## Task
3-class (positive/negative/neutral) **financial** sentiment (investor perspective,
NOT emotional) on short Serbian financial headlines.

## Hard rules
- Test set is HUMAN-labeled only. Never evaluate on LLM-generated labels.
- One fixed prompt template, identical at train / zero-shot baseline / eval.
- Loss on completion only.
- Use current TRL API: SFTConfig (not TrainingArguments), processing_class (not tokenizer=).
- Set tokenizer.pad_token = eos_token.
- Headline metric = macro-F1, not accuracy.
- Production logic in src/, not notebooks. Pin deps.

## Required baselines (same gold test set)
1. Zero-shot Llama 3.2 3B (same prompt).
2. Translation (sr->en) -> ProsusAI/finbert.

## Layout
configs/ data/{raw,interim,processed} src/{collect,label,data,train,eval,publish}
reports/ notebooks/

## Deliverables
GitHub repo · HF dataset repo (+card) · HF model repo (adapter +card) · reports/eval.md

## Definition of done (per phase) — see SPEC.md §8 table.
Success = fine-tuned macro-F1 beats BOTH required baselines on the gold test set.
```
