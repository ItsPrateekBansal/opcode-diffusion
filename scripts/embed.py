"""Step 2 (laptop): split the data and build embeddings.

    python scripts/embed.py --index work/index --out work/emb --mode global --fit train --seeds 0 1 2
    python scripts/embed.py --index work/index --out work/emb --mode per_family --fit all --seeds 0

Writes work/emb/{mode}_{fit}_s{seed}.npz (about 30 MB each).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opdiff.data import TokenStore, make_split  # noqa: E402
from opdiff.embed import build_embeddings  # noqa: E402
from opdiff.io import save_embeddings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="work/index")
ap.add_argument("--out", default="work/emb")
ap.add_argument("--mode", choices=["global", "per_family"], required=True)
ap.add_argument("--fit", choices=["train", "all"], required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--dim", type=int, default=104)
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
ap.add_argument("--max-tokens", type=int, default=0,
                help="use only the first N opcodes of each file to train Word2Vec (0 = all)")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
store = TokenStore(a.index)
families = sorted(store.index.family.unique())
for seed in a.seeds:
    t0 = time.time()
    split = make_split(store.index, seed)
    X, keep, log = build_embeddings(store, split, a.mode, a.fit, seed, dim=a.dim, window=a.window,
                                    epochs=a.epochs, workers=a.workers, max_tokens=a.max_tokens)
    split = split[keep].reset_index(drop=True)
    log.update(seed=seed, counts=split.split.value_counts().to_dict(), seconds=round(time.time() - t0))
    path = os.path.join(a.out, f"{a.mode}_{a.fit}_s{seed}.npz")
    save_embeddings(path, X[keep], split, families, log)
    print(json.dumps({k: log[k] for k in ("mode", "fit", "seed", "counts", "dropped_no_known_tokens",
                                          "dropped_eval_equal_to_train", "seconds")}))
    print("  ->", path)
