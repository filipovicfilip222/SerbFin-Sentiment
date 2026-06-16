"""Print 4-class distribution and 8 examples per class from silver_v2.jsonl."""
import json, random, sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

recs = [json.loads(l) for l in Path("data/interim/silver_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cnt = Counter(r["label"] for r in recs)
total = len(recs)
fin = cnt["positive"] + cnt["negative"] + cnt["neutral"]
irr = cnt["irrelevant"]

print(f"Total labeled : {total}")
print(f"  positive    : {cnt['positive']:5d}  ({cnt['positive']/total:.1%})")
print(f"  negative    : {cnt['negative']:5d}  ({cnt['negative']/total:.1%})")
print(f"  neutral     : {cnt['neutral']:5d}  ({cnt['neutral']/total:.1%})")
print(f"  irrelevant  : {cnt['irrelevant']:5d}  ({cnt['irrelevant']/total:.1%})  <- FILTER")
print(f"  ---")
print(f"  FINANCIAL   : {fin:5d}  ({fin/total:.1%})")
print(f"  IRRELEVANT  : {irr:5d}  ({irr/total:.1%})")

rng = random.Random(42)
for lbl_key in ["positive", "negative", "neutral", "irrelevant"]:
    pool = [r for r in recs if r["label"] == lbl_key]
    rng.shuffle(pool)
    print(f"\n[{lbl_key.upper()}]  n={len(pool)}")
    for r in pool[:8]:
        print(f"  {r['text'][:90]}")
