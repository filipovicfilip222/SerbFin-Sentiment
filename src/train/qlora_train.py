"""
QLoRA fine-tuning — Phase 3 (not yet implemented; Phase 0–1 in progress).

When implementing, follow SPEC.md §5 exactly:

TRL API (≥ 0.16) — CRITICAL, most tutorials use the OLD API:
  - Use SFTConfig (NOT TrainingArguments)
  - Use processing_class=tokenizer (NOT tokenizer=tokenizer)
  - Set tokenizer.pad_token = tokenizer.eos_token (Llama has no pad token)
  - Set completion_only_loss=True (loss on the label token only)

Minimal correct scaffold:

    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**train_cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(**lora_cfg),
        processing_class=tokenizer,     # <-- NOT tokenizer=
    )
    trainer.train()

Prompt template (must be IDENTICAL at train / zero-shot baseline / eval):
    See src/label/silver.py::SYSTEM_PROMPT and build_user_message() for the
    exact strings. The assistant turn is just the label word.
"""

raise NotImplementedError("Training code not yet implemented — complete Phases 1–2 first.")
