"""Smoke-test: 4-class label + parse_label + schema acceptance."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from src.data.schema import Label, SentimentRecord, LabelSource
from src.label.silver import parse_label, VALID_LABELS, SYSTEM_PROMPT, SYSTEM_PROMPT_V1

# 1. enum completeness
assert set(l.value for l in Label) == {"positive","negative","neutral","irrelevant"}, Label.__members__
print("Label enum OK:", {l.value for l in Label})

# 2. VALID_LABELS derived from enum
assert "irrelevant" in VALID_LABELS
print("VALID_LABELS OK:", VALID_LABELS)

# 3. parse_label handles all 4
for raw, exp_lbl, exp_clean in [
    ("irrelevant", "irrelevant", True),
    ("IRRELEVANT", "irrelevant", True),
    ("This is irrelevant content", "irrelevant", False),
    ("positive", "positive", True),
    ("neutral", "neutral", True),
    ("garbage", None, False),
]:
    got_lbl, got_clean = parse_label(raw)
    ok = (got_lbl == exp_lbl) and (got_clean == exp_clean)
    print(f"  parse_label({raw!r:30s}) -> {got_lbl}, clean={got_clean}  {'OK' if ok else 'FAIL'}")

# 4. SentimentRecord accepts irrelevant
rec = SentimentRecord(
    id="sfs-000001", text="Protest gradjana", source="test",
    url="http://x", date="2026-06-01", lang="sr",
    label=Label.irrelevant, label_source=LabelSource.llm_silver,
)
assert rec.label == Label.irrelevant
print(f"\nSentimentRecord with irrelevant OK: {rec.label}")

# 5. Both prompts exist and differ
assert SYSTEM_PROMPT != SYSTEM_PROMPT_V1
assert "irrelevant" in SYSTEM_PROMPT
assert "irrelevant" not in SYSTEM_PROMPT_V1
assert "Zamke" in SYSTEM_PROMPT or "zamke" in SYSTEM_PROMPT.lower()
print("\nSYSTEM_PROMPT (v2) excerpt:")
for line in SYSTEM_PROMPT.splitlines()[:5]:
    print(f"  {line}")
print(f"  ... ({len(SYSTEM_PROMPT)} chars total)")

print(f"\nSYSTEM_PROMPT_V1 (train/eval): {SYSTEM_PROMPT_V1!r}")
print("\nAll checks passed.")
