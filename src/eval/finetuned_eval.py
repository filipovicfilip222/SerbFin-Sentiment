"""
Fine-tuned QLoRA adapter eval (SPEC.md §8, Phase 4 -- the number that decides success).

Loads the base Llama 3.2 3B Instruct model + the LoRA adapter produced by
src/train/qlora_train.py, and scores it on the same gold test set with the
EXACT same prompt as the zero-shot baseline (SYSTEM_PROMPT_V1 +
build_user_message from src/label/silver.py) -- so the only thing that
differs from the zero-shot run is the weights.

Success criterion (SPEC.md §8): this macro-F1 must beat BOTH baselines
already in reports/eval.md (zero-shot Llama and translate+FinBERT).

Requires a CUDA GPU (loads the base model in 4-bit, same as training).

Usage:
    python -m src.eval.finetuned_eval
    python -m src.eval.finetuned_eval --adapter outputs/serbfin-llama-3.2-3b/final_adapter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.label.silver import SYSTEM_PROMPT_V1, build_user_message, parse_label
from src.eval.metrics import load_gold, score, write_report

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_ADAPTER = "outputs/serbfin-llama-3.2-3b/final_adapter"


def run_model(base_model: str, adapter_path: str, gold: list[dict], max_new_tokens: int = 8) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. Run this on the same GPU machine used for training."
        )

    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading base model {base_model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {adapter_path} ...", file=sys.stderr)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    device = next(model.parameters()).device
    results = []

    for rec in tqdm(gold, desc="Fine-tuned adapter"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V1},
            {"role": "user", "content": build_user_message(rec["text"])},
        ]
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
                "pred": pred,
                "raw": raw,
                "clean": is_clean,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuned QLoRA adapter eval")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("reports/predictions_finetuned.jsonl")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/eval.md"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    if not Path(args.adapter).exists():
        raise SystemExit(
            f"Adapter not found at {args.adapter}. Run `python -m src.train.qlora_train` first."
        )

    gold = load_gold(args.test)
    print(f"Loaded {len(gold)} gold records from {args.test}")

    results = run_model(args.base_model, args.adapter, gold, max_new_tokens=args.max_new_tokens)

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions written to {args.predictions}")

    section = score(
        "Fine-tuned QLoRA adapter",
        results,
        notes=f"base={args.base_model}, adapter={args.adapter}, greedy decoding, "
        f"max_new_tokens={args.max_new_tokens}, same SYSTEM_PROMPT_V1 as train/zero-shot",
    )
    write_report([section], args.report)

    print(f"\nmacro-F1        = {section.macro_f1:.4f}")
    print(f"accuracy        = {section.accuracy:.4f}")
    print(f"malformed rate  = {section.malformed_rate:.2%}")
    print(f"Report merged into {args.report}")
    print(
        "\nCompare against reports/eval.md summary table -- fine-tuned macro-F1 must beat "
        "BOTH the zero-shot Llama and Translate+FinBERT baselines (SPEC.md §8)."
    )


if __name__ == "__main__":
    main()
