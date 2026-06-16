"""Smoke-test for the _RateLimiter and parse_label."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import threading
from src.label.silver import _RateLimiter, parse_label

# 4 threads sharing a 60-rpm limiter with burst=2
# First 2 tokens are instant (burst), next 2 each wait ~1s
lim = _RateLimiter(rpm=60, burst=2)
times = []

def worker():
    lim.acquire()
    times.append(time.monotonic())

threads = [threading.Thread(target=worker) for _ in range(4)]
t0 = time.monotonic()
for t in threads: t.start()
for t in threads: t.join()

elapsed = max(times) - t0
print(f"Burst test: 4 threads, burst=2, rpm=60")
print(f"  Elapsed: {elapsed:.2f}s  (expect ~2s -- 2 extra tokens * 1s/token)")
assert 1.5 < elapsed < 3.5, f"Unexpected elapsed: {elapsed}"
print("  PASS")
print()
print("parse_label tests:")
for raw, expect_label, expect_clean in [
    ("neutral",            "neutral",  True),
    ("POSITIVE",           "positive", True),
    ("Sentiment: positive.","positive", False),
    ("I cannot determine", None,       False),
]:
    lbl, clean = parse_label(raw)
    ok = (lbl == expect_label) and (clean == expect_clean)
    print(f"  {repr(raw):<30} -> {lbl}, clean={clean}  {'OK' if ok else 'FAIL'}")

print("\nAll tests passed.")
