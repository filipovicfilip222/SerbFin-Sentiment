# SerbFin-Sentiment — Evaluation

Test set: `data/processed/test.jsonl` (240 headlines, manually labeled by me).

Success criterion (SPEC.md §8): fine-tuned macro-F1 beats BOTH baselines.

## Summary

| model | n | macro-F1 | accuracy | malformed rate |
|---|---|---|---|---|
| Zero-shot Llama 3.2 3B | 240 | 0.6389 | 0.6417 | 0.00% |
| Translate (sr->en) + FinBERT | 240 | 0.5434 | 0.5417 | 0.00% |
| Fine-tuned QLoRA adapter | 240 | 0.7379 | 0.7333 | 0.00% |

<!-- summary: f1=0.6389 malformed=0.0000 -->
## Zero-shot Llama 3.2 3B

- n = 240
- **macro-F1 = 0.6389**
- accuracy = 0.6417
- malformed / unparseable output rate = 0.00%
- notes: model=meta-llama/Llama-3.2-3B-Instruct, greedy decoding, max_new_tokens=8, same SYSTEM_PROMPT_V1 as train/eval

### Per-class

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| positive | 0.800 | 0.450 | 0.576 | 80 |
| negative | 0.740 | 0.704 | 0.722 | 81 |
| neutral | 0.517 | 0.772 | 0.619 | 79 |

### Confusion matrix

| true \ pred | positive | negative | neutral |
|---|---|---|---|
| **positive** | 36 | 8 | 36 |
| **negative** | 3 | 57 | 21 |
| **neutral** | 6 | 12 | 61 |

<!-- summary: f1=0.5434 malformed=0.0000 -->
## Translate (sr->en) + FinBERT

- n = 240
- **macro-F1 = 0.5434**
- accuracy = 0.5417
- malformed / unparseable output rate = 0.00%
- notes: mt=Helsinki-NLP/opus-mt-tc-big-zls-en, finbert=ProsusAI/finbert

### Per-class

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| positive | 0.552 | 0.463 | 0.503 | 80 |
| negative | 0.667 | 0.617 | 0.641 | 81 |
| neutral | 0.439 | 0.544 | 0.486 | 79 |

### Confusion matrix

| true \ pred | positive | negative | neutral |
|---|---|---|---|
| **positive** | 37 | 7 | 36 |
| **negative** | 12 | 50 | 19 |
| **neutral** | 18 | 18 | 43 |

<!-- summary: f1=0.7379 malformed=0.0000 -->
## Fine-tuned QLoRA adapter

- n = 240
- **macro-F1 = 0.7379**
- accuracy = 0.7333
- malformed / unparseable output rate = 0.00%
- notes: base=meta-llama/Llama-3.2-3B-Instruct, adapter=outputs/serbfin-llama-3.2-3b/final_adapter, greedy decoding, max_new_tokens=8, same SYSTEM_PROMPT_V1 as train/zero-shot

### Per-class

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| positive | 0.859 | 0.688 | 0.764 | 80 |
| negative | 0.848 | 0.691 | 0.762 | 81 |
| neutral | 0.591 | 0.823 | 0.688 | 79 |

### Confusion matrix

| true \ pred | positive | negative | neutral |
|---|---|---|---|
| **positive** | 55 | 2 | 23 |
| **negative** | 3 | 56 | 22 |
| **neutral** | 6 | 8 | 65 |
