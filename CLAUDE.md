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
