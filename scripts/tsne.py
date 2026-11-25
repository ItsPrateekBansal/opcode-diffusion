"""t-SNE of real vs synthetic vectors (150 + 150 per family by default).

    python scripts/tsne.py --emb work/emb/global_train_s0.npz \
        --synth work/synth/global_train_s0/diffusion_tmax500_start500.npz --out work/figures
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opdiff.io import load_embeddings, load_synthetic  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--synth", required=True)
ap.add_argument("--out", default="work/figures")
ap.add_argument("--per-family", type=int, default=150)
a = ap.parse_args()

D = load_embeddings(a.emb)
Xs, ys, _ = load_synthetic(a.synth)
rng = np.random.default_rng(0)
R, S = [], []
for c in np.unique(ys):
    r, s = D["Xtr"][D["ytr"] == c], Xs[ys == c]
    R.append(r[rng.permutation(len(r))[: a.per_family]])
    S.append(s[rng.permutation(len(s))[: a.per_family]])
R, S = np.vstack(R), np.vstack(S)
Z = TSNE(n_components=2, random_state=0, init="pca").fit_transform(np.vstack([R, S]))
plt.figure(figsize=(6, 5))
plt.scatter(*Z[: len(R)].T, s=4, c="black", label="real")
plt.scatter(*Z[len(R):].T, s=4, c="0.6", label="synthetic")
plt.legend(); plt.xticks([]); plt.yticks([])
name = os.path.splitext(os.path.basename(a.synth))[0]
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"tsne_{os.path.splitext(os.path.basename(a.emb))[0]}_{name}.png")
plt.tight_layout(); plt.savefig(path, dpi=200)
print("->", path)
