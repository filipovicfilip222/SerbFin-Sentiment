"""
Push the fine-tuned LoRA adapter to the HF Hub (SPEC.md §7.2, §7.3).

Uploads the adapter produced by src/train/qlora_train.py as-is (no base-model
merge -- ships as a small LoRA adapter repo, loaded with PEFT on top of the
gated meta-llama/Llama-3.2-3B-Instruct base). Builds a model card pulling the
eval summary table straight out of reports/eval.md (not hand-typed), plus the
Llama 3.2 Community License attribution required by its terms.

Requires HF_TOKEN in .env with WRITE access (a "Read"-only token will fail
with a 403). No GPU needed -- this only uploads local files.

Usage:
    python -m src.publish.push_model
    python -m src.publish.push_model --repo-id filipovicfilip222/serbfin-sentiment-llama-3.2-3b
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REPO_ID = "filipovicfilip222/serbfin-sentiment-llama-3.2-3b"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def parse_summary_table(eval_md: Path) -> str:
    """Pull the '## Summary' markdown table out of reports/eval.md verbatim."""
    if not eval_md.exists():
        return "*(reports/eval.md not found -- run the eval scripts first)*"
    text = eval_md.read_text(encoding="utf-8")
    m = re.search(r"## Summary\n\n(.*?)\n\n", text, re.DOTALL)
    return m.group(1) if m else "*(could not parse summary table from reports/eval.md)*"


def build_card(repo_id: str, adapter_dir: Path, lora_cfg: dict, train_cfg: dict, summary_table: str) -> str:
    dataset_owner = repo_id.rsplit("/", 1)[0]
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
license: llama3.2
language:
- sr
tags:
- financial-sentiment
- serbian
- lora
- qlora
pipeline_tag: text-generation
datasets:
- {dataset_owner}/serbfin-sentiment
---

# SerbFin-Sentiment -- Llama 3.2 3B (LoRA adapter)

**Built with Llama.** QLoRA (4-bit NF4) fine-tuned adapter for [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}),
for 3-class **Serbian financial sentiment** classification (`positive` / `negative` / `neutral`)
on short financial news headlines, judged from an **investor / market perspective** (not
emotional sentiment).

First publicly available Serbian-language financial sentiment model.

## Results

Evaluated on a 240-headline **human-labeled** gold test set (never used for training), same
fixed prompt at zero-shot baseline / training / eval time. Full per-class breakdown and
confusion matrices: `reports/eval.md` in the
[GitHub repo](https://github.com/filipovicfilip222-alt/SerbFin-Sentiment).

{summary_table}

Fine-tuned macro-F1 beats both required baselines (zero-shot Llama 3.2 3B, and a
translate-to-English + [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) pipeline).

## Training data

[{dataset_owner}/serbfin-sentiment](https://huggingface.co/datasets/{dataset_owner}/serbfin-sentiment) --
`train` + `validation` splits (silver-labeled by Claude Haiku). The `test` split used for the
numbers above is human-gold-labeled and was never seen during training.

## Hyperparameters

QLoRA (`configs/lora.yaml`):
```yaml
{yaml.dump(lora_cfg, sort_keys=False)}```

Training (`configs/train.yaml`):
```yaml
{yaml.dump(train_cfg, sort_keys=False)}```

## Prompt template

Identical at train / zero-shot baseline / eval time (`src/label/silver.py::SYSTEM_PROMPT_V1`):
```
<system>
Ti si klasifikator finansijskog sentimenta. Za dati naslov vesti odredi sentiment sa
perspektive investitora.
Odgovori isključivo jednom rečju: positive, negative ili neutral.
</system>
<user>
Naslov: {{text}}
Sentiment:
</user>
<assistant>
{{label}}
```

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model = "{BASE_MODEL}"
adapter_id = "{repo_id}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

messages = [
    {{"role": "system", "content": "Ti si klasifikator finansijskog sentimenta. Za dati naslov vesti odredi sentiment sa perspektive investitora.\\nOdgovori isključivo jednom rečju: positive, negative ili neutral."}},
    {{"role": "user", "content": "Naslov: NIS objavio rekordnu dobit u prvom kvartalu\\nSentiment:"}},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Reproduction

```bash
git clone https://github.com/filipovicfilip222-alt/SerbFin-Sentiment
cd SerbFin-Sentiment && pip install -e .
python -m src.data.split --train-val-only
python -m src.train.qlora_train      # requires a CUDA GPU
python -m src.eval.finetuned_eval
```

## Intended use & limitations

Financial-news-headline sentiment, Serbian language, investor/market framing. Not intended for
emotional sentiment, social media text, or languages other than Serbian. The `positive`/`neutral`
boundary is the most common source of confusion -- see the confusion matrix in `reports/eval.md`.
Class balance in training data skews `neutral`.

## License

Adapter weights: **Llama 3.2 Community License** (this model is a derivative of Llama 3.2 3B
Instruct -- see [the license](https://www.llama.com/llama3_2/license/) and
[acceptable use policy](https://www.llama.com/llama3_2/use-policy)). "Llama 3.2 is licensed under
the Llama 3.2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved."
Built with Llama.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Push fine-tuned LoRA adapter to HF Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("outputs/serbfin-llama-3.2-3b/final_adapter")
    )
    parser.add_argument("--lora-config", type=Path, default=Path("configs/lora.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--eval-report", type=Path, default=Path("reports/eval.md"))
    parser.add_argument("--private", action="store_true", help="Push as a private repo")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN not set in .env")

    if not args.adapter_dir.exists():
        raise SystemExit(
            f"{args.adapter_dir} not found. Run `python -m src.train.qlora_train` first."
        )

    from huggingface_hub import HfApi, create_repo

    lora_cfg = yaml.safe_load(args.lora_config.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    summary_table = parse_summary_table(args.eval_report)

    card = build_card(args.repo_id, args.adapter_dir, lora_cfg, train_cfg, summary_table)
    card_path = args.adapter_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Model card written to {card_path}")

    print(f"Creating/using repo {args.repo_id} (model)...")
    create_repo(args.repo_id, token=hf_token, repo_type="model", exist_ok=True, private=args.private)

    print(f"Uploading {args.adapter_dir} ...")
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=str(args.adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        token=hf_token,
    )

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
