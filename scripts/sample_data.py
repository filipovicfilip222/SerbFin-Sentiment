"""Sample raw data to identify junk patterns."""
import json, re, sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

lines = [l for l in Path("data/raw/headlines.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
recs  = [json.loads(l) for l in lines]
texts = [r["text"] for r in recs]

print(f"Total records: {len(recs)}")

wc = Counter(len(t.split()) for t in texts)
print("\nWord-count distribution:")
for n in sorted(wc):
    bar = "#" * min(wc[n], 40)
    print(f"  {n:3d}: {wc[n]:4d}  {bar}")

short = [t for t in texts if len(t.split()) < 5]
print(f"\nShort (<5 words): {len(short)}")
for t in short: print(f"  {repr(t)}")

long_ = [t for t in texts if len(t.split()) > 30]
print(f"\nLong (>30 words): {len(long_)}")
for t in long_[:10]: print(f"  {repr(t[:120])}")

# Suffix patterns
suffix_re = re.compile(r"[\s\-|]+(?:Biznis\.rs|B92|eKapija|Nova Ekonomija|novaekonomija|Kamatica)\s*$", re.I)
suffix = [t for t in texts if suffix_re.search(t)]
print(f"\nWith site-name suffix: {len(suffix)}")
for t in suffix[:10]: print(f"  {repr(t[:100])}")

# Junk patterns
junk_re = re.compile(r"(?:404|Page not found|naslovna|kolačić|cookie|Prihvati|consent|Nastavi)", re.I)
junk = [t for t in texts if junk_re.search(t)]
print(f"\nJunk-pattern matches: {len(junk)}")
for t in junk: print(f"  {repr(t[:100])}")

# Near-duplicates by first 50 chars
prefixes = Counter(t[:50].lower().strip() for t in texts)
common = [(p, n) for p, n in prefixes.most_common(30) if n > 1]
print(f"\nNear-duplicate prefixes (first 50 chars, count>1): {len(common)}")
for p, n in common[:20]: print(f"  {n}x  {repr(p)}")
