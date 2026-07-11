"""
Zero-shot Llama 3.2 3B Instruct baseline (SPEC.md §8, Phase 4 — required baseline #1).

Uses the EXACT same prompt as training and silver labeling:
SYSTEM_PROMPT_V1 + build_user_message() from src/label/silver.py. This is the
whole point of the baseline: the only thing that changes vs. the fine-tuned
model is the weights, not the prompt.

Requires a local GPU (or enough RAM for CPU inference, which will be slow) and
either a local copy of the model or `huggingface-cli login` / HF_TOKEN in .env
with access to meta-llama/Llama-3.2-3B-Instruct (gated repo — request access
on the HF model page first).

Usage:
    python -m src.eval.zero_shot_llama
    python -m src.eval.zero_shot_llama --model meta-llama/Llama-3.2-3B-Instruct
    python -m src.eval.zero_shot_llama --test data/processed/test.jsonl \
        --predictions reports/predictions_zeroshot.jsonl --report reports/eval.md

Writes:
    reports/predictions_zeroshot.jsonl   -- one prediction per gold record
    reports/eval.md                      -- merged report (see src/eval/metrics.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Reuse the exact prompt + label parser used for silver labeling, so the
# zero-shot baseline and the fine-tuned model are judged on an identical
# instruction. Do NOT redefine these strings here.
from src.label.silver import SYSTEM_PROMPT_V1, build_user_message, parse_label
from src.eval.metrics import load_gold, score, write_report

load_dotenv()

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def run_model(model_name: str, gold: list[dict], max_new_tokens: int = 8) -> list[dict]:
    """
    Run the local Llama model over every gold headline with the fixed prompt.

    Returns a list of {"id", "text", "label", "pred", "raw"} dicts, same order
    as `gold`, ready to hand to src.eval.metrics.score().
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading {model_name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    results = []

    for rec in tqdm(gold, desc="Zero-shot Llama"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V1},
            {"role": "user", "content": build_user_message(rec["text"])},
        ]
        # return_dict=True is required explicitly: newer transformers versions
        # (5.x) no longer return a bare tensor from apply_chat_template even
        # with return_tensors="pt" -- passing **inputs to generate() also
        # supplies attention_mask, which is the version-safe way to call it.
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_tokens = output[0][inputs["input_ids"].shape[-1]:]
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred, is_clean = parse_label(raw)

        results.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "label": rec["label"],
                "pred": pred,  # None if unparseable -- metrics.score() treats that as malformed
                "raw": raw,
                "clean": is_clean,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot Llama 3.2 3B baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("reports/predictions_zeroshot.jsonl")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/eval.md"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    gold = load_gold(args.test)
    print(f"Loaded {len(gold)} gold records from {args.test}")

    results = run_model(args.model, gold, max_new_tokens=args.max_new_tokens)

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions written to {args.predictions}")

    section = score(
        "Zero-shot Llama 3.2 3B",
        results,
        notes=f"model={args.model}, greedy decoding, max_new_tokens={args.max_new_tokens}, "
        "same SYSTEM_PROMPT_V1 as train/eval",
    )
    write_report([section], args.report)

    print(f"\nmacro-F1        = {section.macro_f1:.4f}")
    print(f"accuracy        = {section.accuracy:.4f}")
    print(f"malformed rate  = {section.malformed_rate:.2%}")
    print(f"Report merged into {args.report}")


if __name__ == "__main__":
    main()
