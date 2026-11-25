"""Step 1 (laptop): read the corpus once and cache the top-25 families.

    python scripts/index_data.py --src path/to/corpus.zip --out work/index
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opdiff.data import build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="zip file or extracted folder")
ap.add_argument("--out", default="work/index")
ap.add_argument("--top-k", type=int, default=25)
ap.add_argument("--tokenize", choices=["whitespace", "first"], default="whitespace")
ap.add_argument("--token-regex", default=None)
ap.add_argument("--keep-first", type=int, default=10000, help="opcodes per file kept for Word2Vec")
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()

t0 = time.time()
s = build_index(a.src, a.out, a.top_k, a.tokenize, a.token_regex, a.keep_first, a.workers)
print(json.dumps({k: v for k, v in s.items() if k != "duplicates"}, indent=2))
print("\nduplicates per family (files -> unique):")
for f, d in s["duplicates"]["per_family"].items():
    print(f"  {f:<20} {d['files']:>7,} -> {d['unique']:,}")
print(f"hashes found in more than one family: {s['duplicates']['hashes_in_more_than_one_family']}")
print(f"done in {time.time() - t0:.0f}s -> {a.out}")
