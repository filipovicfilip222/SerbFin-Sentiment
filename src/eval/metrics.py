"""
Shared evaluation utilities for all baselines + the fine-tuned model
(SPEC.md §6-7 — Phase 4).

Every baseline (zero-shot Llama, translate+finbert, fine-tuned adapter) writes
predictions in the same shape and calls the functions here, so the numbers in
reports/eval.md are computed identically for all of them:

    {"id": ..., "text": ..., "label": <gold>, "pred": <predicted-or-None>, "raw": <raw model output>}

Usage (as a library, from a baseline script):

    from src.eval.metrics import load_gold, score, write_report

    gold = load_gold(Path("data/processed/test.jsonl"))
    preds = [...]  # list[dict] as above, one per gold record, same order/ids
    section = score("Zero-shot Llama 3.2 3B", preds)
    write_report([section], Path("reports/eval.md"))

write_report() merges into any existing reports/eval.md rather than
overwriting it, so baselines and the fine-tuned model can each be run
independently (possibly on different days/machines) and still end up in one
combined report with an up-to-date summary table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

LABELS = ("positive", "negative", "neutral")

# Preferred ordering in the summary table / report body.
NAME_ORDER = [
    "Zero-shot Llama 3.2 3B",
    "Translate (sr->en) + FinBERT",
    "Fine-tuned QLoRA adapter",
]


def load_gold(path: Path) -> list[dict]:
    """Load the human-labeled gold test set. Fails loudly if not human-labeled."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label_source") != "human":
                raise ValueError(
                    f"Gold test set contains a non-human label (id={rec.get('id')}, "
                    f"label_source={rec.get('label_source')!r}). "
                    "SPEC.md §4.4: the test set MUST be human-labeled. Refusing to score."
                )
            if rec.get("label") not in LABELS:
                raise ValueError(f"Unexpected label {rec.get('label')!r} for id={rec.get('id')}")
            records.append(rec)
    return records


@dataclass
class ReportSection:
    name: str
    n: int
    macro_f1: float
    accuracy: float
    per_class: dict  # label -> {precision, recall, f1, support}
    confusion: list  # 3x3, rows=true, cols=pred, order=LABELS
    malformed_rate: float
    malformed_examples: list = field(default_factory=list)
    notes: str = ""


def score(name: str, preds: list[dict], *, notes: str = "") -> ReportSection:
    """
    Compute macro-F1 + confusion matrix from a list of prediction records.

    Each record must have "label" (gold, one of LABELS) and "pred" (model's
    predicted label, one of LABELS, or None/other string if malformed/unparseable
    -- those never count as correct but still consume a gold example, i.e. they
    always hurt recall on the true class).
    """
    if not preds:
        raise ValueError("score() called with an empty prediction list")

    y_true = [r["label"] for r in preds]
    y_pred = [r["pred"] if r.get("pred") in LABELS else "__malformed__" for r in preds]

    malformed = [r for r in preds if r.get("pred") not in LABELS]
    malformed_rate = len(malformed) / len(preds)

    macro_f1 = f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(preds)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    per_class = {
        lab: {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }
        for i, lab in enumerate(LABELS)
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()

    return ReportSection(
        name=name,
        n=len(preds),
        macro_f1=float(macro_f1),
        accuracy=float(accuracy),
        per_class=per_class,
        confusion=cm,
        malformed_rate=malformed_rate,
        malformed_examples=[
            {"id": r.get("id"), "text": r.get("text", "")[:80], "raw": r.get("raw", "")}
            for r in malformed[:5]
        ],
        notes=notes,
    )


def _confusion_table(section: ReportSection) -> str:
    header = "| true \\ pred | " + " | ".join(LABELS) + " |"
    sep = "|---|" + "---|" * len(LABELS)
    rows = []
    for i, lab in enumerate(LABELS):
        row = " | ".join(str(section.confusion[i][j]) for j in range(len(LABELS)))
        rows.append(f"| **{lab}** | {row} |")
    return "\n".join([header, sep, *rows])


def _per_class_table(section: ReportSection) -> str:
    header = "| class | precision | recall | f1 | support |"
    sep = "|---|---|---|---|---|"
    rows = []
    for lab in LABELS:
        c = section.per_class[lab]
        rows.append(
            f"| {lab} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |"
        )
    return "\n".join([header, sep, *rows])


def render_section_md(section: ReportSection) -> str:
    """Render one section's body (heading + stats + tables), no leading summary comment."""
    parts = [
        f"## {section.name}",
        "",
        f"- n = {section.n}",
        f"- **macro-F1 = {section.macro_f1:.4f}**",
        f"- accuracy = {section.accuracy:.4f}",
        f"- malformed / unparseable output rate = {section.malformed_rate:.2%}",
    ]
    if section.notes:
        parts.append(f"- notes: {section.notes}")
    parts += ["", "### Per-class", "", _per_class_table(section)]
    parts += ["", "### Confusion matrix", "", _confusion_table(section)]
    if section.malformed_examples:
        parts += ["", "### Malformed output examples"]
        for ex in section.malformed_examples:
            parts.append(f"- `{ex['id']}` \"{ex['text']}\" -> raw: `{ex['raw']!r}`")
    parts.append("")
    return "\n".join(parts)


def _parse_existing_sections(path: Path) -> dict[str, tuple[float, float, str]]:
    """
    Read a previously-written eval.md and recover, per section name:
    (macro_f1, malformed_rate, rendered_body).

    Each section is stored with a leading `<!-- summary: f1=.. malformed=.. -->`
    comment specifically so this round-trip doesn't need to re-parse tables.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    out: dict[str, tuple[float, float, str]] = {}
    blocks = text.split("\n<!-- summary:")
    for block in blocks[1:]:
        comment_end = block.find("-->")
        if comment_end == -1:
            continue
        payload = block[:comment_end].strip()
        body = block[comment_end + 3:].lstrip("\n")
        try:
            f1_str, mal_str = payload.split(" ")
            f1v = float(f1_str.split("=")[1])
            malv = float(mal_str.split("=")[1])
        except Exception:
            continue
        name = body.split("\n", 1)[0].lstrip("#").strip()
        out[name] = (f1v, malv, body.rstrip())
    return out


def write_report(
    sections: list[ReportSection], path: Path, *, title: str = "SerbFin-Sentiment — Evaluation"
) -> None:
    """
    Write/merge sections into reports/eval.md.

    Sections passed here overwrite any previously-saved section with the same
    name; sections from a prior run that aren't being re-scored now are kept
    as-is. The summary table at the top always reflects the merged set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, tuple[float, float, str]] = _parse_existing_sections(path)
    for s in sections:
        merged[s.name] = (s.macro_f1, s.malformed_rate, render_section_md(s))

    ordered_names = [n for n in NAME_ORDER if n in merged] + [
        n for n in merged if n not in NAME_ORDER
    ]

    lines = [
        f"# {title}",
        "",
        "Gold test set: `data/processed/test.jsonl` (240 human-labeled headlines).",
        "",
        "Success criterion (SPEC.md §8): fine-tuned macro-F1 beats BOTH baselines.",
        "",
        "## Summary",
        "",
        "| model | macro-F1 | malformed rate |",
        "|---|---|---|",
    ]
    for name in ordered_names:
        f1v, malv, _ = merged[name]
        lines.append(f"| {name} | {f1v:.4f} | {malv:.2%} |")
    lines.append("")

    for name in ordered_names:
        f1v, malv, body = merged[name]
        lines.append(f"<!-- summary: f1={f1v:.4f} malformed={malv:.4f} -->")
        lines.append(body)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
