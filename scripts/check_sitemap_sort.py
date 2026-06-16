"""Quick diagnostic: compare old vs new child-sitemap selection for Biznis.rs."""
import re
import requests
import xml.etree.ElementTree as ET
from src.collect.sitemap import _t, _child_text, _numeric_key, _NS_SM

UA = "SerbFinSentimentCollector/1.0 (diagnostic)"
r = requests.get("https://biznis.rs/sitemap.xml", headers={"User-Agent": UA}, timeout=20)
root = ET.fromstring(r.content)

entries = []
for sm in root.findall(_t(_NS_SM, "sitemap")):
    loc = _child_text(sm, _t(_NS_SM, "loc"))
    lm  = _child_text(sm, _t(_NS_SM, "lastmod"))
    if re.search("post-sitemap", loc):
        entries.append((loc, lm))

print(f"Total post-sitemaps in index: {len(entries)}")
print()

by_num = sorted(entries, key=lambda e: _numeric_key(e[0]), reverse=True)[:8]
print("=== Old sort (numeric desc) -- first 8 ===")
for loc, lm in by_num:
    print(f"  {lm[:10]}  {loc.rsplit('/', 1)[-1]}")

by_lm = sorted(entries, key=lambda e: (e[1], _numeric_key(e[0])), reverse=True)[:8]
print()
print("=== New sort (lastmod desc) -- first 8 ===")
for loc, lm in by_lm:
    print(f"  {lm[:10]}  {loc.rsplit('/', 1)[-1]}")
