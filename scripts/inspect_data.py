"""Look inside the corpus before indexing: folder layout and file format.

    python scripts/inspect_data.py --src path/to/VirusShare_opcodes_extracted_v077.zip
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opdiff.data import Corpus, family_of, tokenize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--n-sample", type=int, default=30)
a = ap.parse_args()

corpus = Corpus(a.src)
paths = list(corpus.paths())
fams = Counter(family_of(p) for p in paths)
print(f"files: {len(paths):,}   family folders: {len(fams):,}")
print("example paths:", *paths[:3], sep="\n  ")
print("\nlargest folders:")
for f, c in fams.most_common(30):
    print(f"  {c:>7,}  {f}")

print("\nfirst 300 characters of one file:")
print(corpus.read(paths[0], limit=300).decode("utf-8", "ignore"))

ws, first = Counter(), Counter()
for p in paths[:: max(1, len(paths) // a.n_sample)][: a.n_sample]:
    raw = corpus.read(p, limit=2_000_000)  # first 2 MB is enough to see the format
    ws.update(tokenize(raw, "whitespace"))
    first.update(tokenize(raw, "first"))
print(f"\nsample of {a.n_sample} files")
print(f"  whitespace tokens: {sum(ws.values()):,} total, {len(ws):,} distinct; top: {ws.most_common(12)}")
print(f"  first-per-line:    {sum(first.values()):,} total, {len(first):,} distinct; top: {first.most_common(12)}")
print("\nIf the whitespace top tokens include registers (eax, ebp) or hex values,")
print("the files hold full instructions: index with --tokenize first.")
