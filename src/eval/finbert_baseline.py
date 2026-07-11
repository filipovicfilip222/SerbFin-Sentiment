"""
Translate (sr -> en) + ProsusAI/finbert baseline (SPEC.md §8, Phase 4 —
required baseline #2).

Two-stage pipeline, both stages local/offline once downloaded:
  1. Translate each Serbian headline to English with an OPUS-MT model.
     Default: Helsinki-NLP/opus-mt-tc-big-zls-en -- "zls" = South Slavic
     languages (bs/hr/mk/sl/sr...), trained many-to-one into English, no
     language-prefix token needed. There is no standalone opus-mt-sr-en model;
     this is the standard OPUS-MT choice for Serbian.
  2. Classify the translated English text with ProsusAI/finbert (financial
     sentiment BERT, outputs positive/negative/neutral directly).

This baseline does NOT use SYSTEM_PROMPT_V1 (finbert is a classifier head, not
an instruction-following LLM) -- it's included for a like-for-like macro-F1
comparison against the fine-tuned model and the zero-shot LLM baseline, all on
the same gold test set.

Usage:
    python -m src.eval.finbert_baseline
    python -m src.eval.finbert_baseline --mt-model Helsinki-NLP/opus-mt-tc-big-zls-en \
        --finbert-model ProsusAI/finbert --batch-size 16
    python -m src.eval.finbert_baseline --no-cache   # force re-translation

Writes:
    reports/predictions_finbert.jsonl   -- id, text, translated text, gold, pred
    reports/translations_cache.jsonl    -- id -> translated text (reused on rerun,
                                            so a crash/retry during the FinBERT
                                            download doesn't force re-translating
                                            all 240 headlines again)
    reports/eval.md                     -- merged report (see src/eval/metrics.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.eval.metrics import load_gold, score, write_report

load_dotenv()

DEFAULT_MT_MODEL = "Helsinki-NLP/opus-mt-tc-big-zls-en"
DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"


def load_translation_cache(path: Path, mt_model_name: str) -> dict[str, str]:
    """id -> translated text, but only entries from a matching --mt-model."""
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("mt_model") == mt_model_name:
                cache[rec["id"]] = rec["translated"]
    return cache


def save_translation_cache(path: Path, mt_model_name: str, gold: list[dict], translations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for g, t in zip(gold, translations):
            fh.write(json.dumps({"id": g["id"], "mt_model": mt_model_name, "translated": t}, ensure_ascii=False) + "\n")


def translate_batch(texts: list[str], mt_model_name: str, batch_size: int, device) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    model = MarianMTModel.from_pretrained(mt_model_name).to(device)
    model.eval()

    translations: list[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating sr->en"):
        chunk = texts[i : i + batch_size]
        batch = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=128)
        translations.extend(tokenizer.decode(t, skip_special_tokens=True) for t in out)

    del model
    return translations


def classify_batch(
    texts: list[str], finbert_model_name: str, batch_size: int, device
) -> list[tuple[str, dict]]:
    """Returns list of (predicted_label_lowercase, raw_scores_dict)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name).to(device)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    results: list[tuple[str, dict]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT classify"):
        chunk = texts[i : i + batch_size]
        batch = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
        for row in probs:
            scores = {id2label[j]: float(row[j]) for j in range(len(row))}
            pred = max(scores, key=scores.get)
            results.append((pred, scores))

    del model
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate(sr->en) + FinBERT baseline")
    parser.add_argument("--mt-model", default=DEFAULT_MT_MODEL)
    parser.add_argument("--finbert-model", default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("reports/predictions_finbert.jsonl")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/eval.md"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--translation-cache", type=Path, default=Path("reports/translations_cache.jsonl")
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore any cached translations, re-translate all"
    )
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gold = load_gold(args.test)
    print(f"Loaded {len(gold)} gold records from {args.test}")

    cache = {} if args.no_cache else load_translation_cache(args.translation_cache, args.mt_model)
    if cache and all(g["id"] in cache for g in gold):
        print(f"Reusing {len(gold)} cached translations from {args.translation_cache} "
              f"(mt_model={args.mt_model}) -- skipping translation step.")
        translations = [cache[g["id"]] for g in gold]
    else:
        texts = [g["text"] for g in gold]
        translations = translate_batch(texts, args.mt_model, args.batch_size, device)
        save_translation_cache(args.translation_cache, args.mt_model, gold, translations)
        print(f"Translations cached to {args.translation_cache}")

    classified = classify_batch(translations, args.finbert_model, args.batch_size, device)

    results = []
    for g, en_text, (pred, scores) in zip(gold, translations, classified):
        # ProsusAI/finbert's label set is exactly {positive, negative, neutral},
        # so no mapping is needed -- guard anyway in case a different finbert
        # checkpoint is passed via --finbert-model.
        from src.eval.metrics import LABELS

        safe_pred = pred if pred in LABELS else None
        results.append(
            {
                "id": g["id"],
                "text": g["text"],
                "translated": en_text,
                "label": g["label"],
                "pred": safe_pred,
                "raw": pred,
                "scores": scores,
            }
        )

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions written to {args.predictions}")

    section = score(
        "Translate (sr->en) + FinBERT",
        results,
        notes=f"mt={args.mt_model}, finbert={args.finbert_model}",
    )
    write_report([section], args.report)

    print(f"\nmacro-F1        = {section.macro_f1:.4f}")
    print(f"accuracy        = {section.accuracy:.4f}")
    print(f"malformed rate  = {section.malformed_rate:.2%}")
    print(f"Report merged into {args.report}")


if __name__ == "__main__":
    main()
