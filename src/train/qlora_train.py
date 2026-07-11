"""
QLoRA fine-tuning for SerbFin-Sentiment (SPEC.md §5 -- Phase 3).

Trains a LoRA adapter on top of 4-bit-quantized Llama 3.2 3B Instruct using
the EXACT same prompt as the zero-shot baseline and silver labeling
(src/label/silver.py::SYSTEM_PROMPT_V1 + build_user_message), so the only
thing that differs between zero-shot and fine-tuned eval is the weights.

Hard rules (SPEC.md / CLAUDE.md -- do not deviate):
  - SFTConfig (NOT TrainingArguments), processing_class= (NOT tokenizer=)
  - tokenizer.pad_token = tokenizer.eos_token (Llama has no pad token)
  - completion_only_loss=True -- loss computed on the label token(s) only,
    never on the prompt. Implemented via TRL's prompt-completion dataset
    format: two columns, "prompt" and "completion" (both raw strings, not
    pre-tokenized) -- TRL tokenizes and masks the prompt internally.
  - Same fixed prompt template at train / zero-shot baseline / eval time.

Requires a CUDA GPU (bitsandbytes 4-bit has no CPU/AMD path). Run this on a
rented GPU (e.g. a RunPod pod), not locally.

Usage:
    # 1. Build train/val splits from the silver-labeled pool (CPU, run locally first):
    python -m src.data.split --train-val-only --stats

    # 2. On the GPU machine:
    python -m src.train.qlora_train
    python -m src.train.qlora_train --train data/processed/train.jsonl --val data/processed/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.label.silver import SYSTEM_PROMPT_V1, build_user_message

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run `python -m src.data.split --train-val-only` first."
        )
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_prompt_completion(records: list[dict], tokenizer) -> Dataset:
    """
    Build a TRL prompt-completion dataset: "prompt" is the fully chat-templated
    prefix (system + user turn, generation prompt appended, same as the
    zero-shot baseline), "completion" is just the label word. TRL tokenizes
    both internally and, with completion_only_loss=True, masks the prompt so
    loss is computed on the completion tokens only.
    """
    prompts, completions = [], []
    for rec in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V1},
            {"role": "user", "content": build_user_message(rec["text"])},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)
        completions.append(" " + rec["label"])
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Llama 3.2 3B for SerbFin-Sentiment")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--lora-config", type=Path, default=Path("configs/lora.yaml"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. QLoRA (bitsandbytes 4-bit) requires CUDA -- "
            "run this on a GPU machine (e.g. a RunPod pod), not locally on CPU/AMD."
        )

    train_records = load_records(args.train)
    val_records = load_records(args.val)
    print(f"train: {len(train_records)}  val: {len(val_records)}")

    train_cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    lora_cfg_dict = yaml.safe_load(args.lora_config.read_text(encoding="utf-8"))

    print(f"Loading tokenizer + base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # hard rule -- Llama has no pad token

    train_ds = to_prompt_completion(train_records, tokenizer)
    val_ds = to_prompt_completion(val_records, tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(**lora_cfg_dict)
    sft_config = SFTConfig(**train_cfg)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,  # hard rule -- NOT tokenizer=
    )

    print("\nStarting training...")
    trainer.train()

    final_dir = Path(train_cfg["output_dir"]) / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nAdapter + tokenizer saved to {final_dir}")
    print("Next: python -m src.eval.finetuned_eval  (not yet written)")


if __name__ == "__main__":
    main()
