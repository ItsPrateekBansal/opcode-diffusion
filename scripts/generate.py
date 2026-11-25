"""Step 3 (GPU): make synthetic vectors for every family.

    # paper's diffusion: trained to T/2, sampling starts from real samples at T/2
    python scripts/generate.py --emb work/emb/global_train_s0.npz --method diffusion \
        --train-tmax 500 --start-t 500
    # start-point sweep: one model trained on all timesteps, four start points
    python scripts/generate.py --emb work/emb/global_train_s0.npz --method diffusion \
        --train-tmax 1000 --start-t 250 500 750 1000
    python scripts/generate.py --emb work/emb/global_train_s0.npz --method gan
    python scripts/generate.py --emb work/emb/global_train_s0.npz --method wgangp
    python scripts/generate.py --emb work/emb/global_train_s0.npz --method smote   # also adasyn, noise

Writes work/synth/<emb name>/<method>.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opdiff import baselines, models  # noqa: E402
from opdiff.io import load_embeddings, save_synthetic  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--method", required=True, choices=["gan", "wgangp", "diffusion", "smote", "adasyn", "noise"])
ap.add_argument("--out", default="work/synth")
ap.add_argument("--n-per-family", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--fit", choices=["train", "all"], default="train",
                help="train: generators see training rows only. all: also val/test (the earlier baseline)")
ap.add_argument("--train-tmax", type=int, default=500)
ap.add_argument("--start-t", type=int, nargs="+", default=[500])
ap.add_argument("--noise-sigma", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=None, help="default: the embedding's seed")
ap.add_argument("--families", type=int, default=0, help="only the first N families (quick tests)")
a = ap.parse_args()

D = load_embeddings(a.emb)
seed = D["log"]["seed"] if a.seed is None else a.seed
X, y = (D["Xtr"], D["ytr"]) if a.fit == "train" else (D["X_all"], D["y_all"])
classes = np.unique(D["ytr"])
if a.families:
    classes = classes[: a.families]
device = "cuda" if torch.cuda.is_available() else "cpu"
stem = os.path.splitext(os.path.basename(a.emb))[0]
out_dir = os.path.join(a.out, stem)
os.makedirs(out_dir, exist_ok=True)
tag_fit = "" if a.fit == "train" else "_fitall"
print(f"{a.method} on {stem} ({len(classes)} families, device={device}, fit={a.fit})")

meta = {"method": a.method, "emb": stem, "seed": seed, "fit": a.fit, "n_per_family": a.n_per_family,
        "epochs": a.epochs, "seconds": {}}

if a.method in ("smote", "adasyn", "noise"):
    m = np.isin(y, classes)
    t0 = time.time()
    if a.method == "noise":
        Xs, ys, log = baselines.gaussian_noise(X[m], y[m], a.n_per_family, a.noise_sigma, seed)
    else:
        Xs, ys, log = baselines.oversample(X[m], y[m], a.n_per_family, a.method, seed)
    meta.update(log={str(k): v for k, v in log.items()}, seconds=round(time.time() - t0))
    save_synthetic(os.path.join(out_dir, f"{a.method}{tag_fit}.npz"), Xs, ys, meta)
    print(f"  {len(Xs):,} vectors, {meta['seconds']}s")
    sys.exit()

cfg = {"epochs": a.epochs, "train_tmax": a.train_tmax}
outs = {}  # name -> list of arrays
for k, c in enumerate(classes):
    Xc = X[y == c]
    if len(Xc) < 2:
        Xc = np.repeat(Xc, 2, axis=0)  # BatchNorm needs at least two rows
    t0 = time.time()
    s = seed * 1000 + int(c)
    if a.method == "gan":
        G = models.train_gan(Xc, cfg, device, s)
        outs.setdefault("gan", []).append(models.sample_gan(G, a.n_per_family, device))
    elif a.method == "wgangp":
        G = models.train_wgangp(Xc, cfg, device, s)
        outs.setdefault("wgangp", []).append(models.sample_wgangp(G, a.n_per_family, device))
    else:
        diff = models.Diffusion(device=device)
        net = diff.train(Xc, cfg, s)
        for st in a.start_t:
            if st > a.train_tmax:
                print(f"  warning: start_t {st} > train_tmax {a.train_tmax}; the model never saw that noise level")
            Xs_c, _ = diff.sample(net, Xc, a.n_per_family, st, s)
            outs.setdefault(f"diffusion_tmax{a.train_tmax}_start{st}", []).append(Xs_c)
    meta["seconds"][D["families"][c]] = round(time.time() - t0, 1)
    print(f"  [{k + 1}/{len(classes)}] {D['families'][c]:<18} rows={len(Xc):>6,}  {time.time() - t0:.0f}s", flush=True)

ys = np.repeat(classes, a.n_per_family)
for name, parts in outs.items():
    path = os.path.join(out_dir, f"{name}{tag_fit}.npz")
    save_synthetic(path, np.concatenate(parts), ys, {**meta, "name": name})
    print("  ->", path)
