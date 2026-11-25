"""Step 4: real-only baselines and the six metrics for each synthetic set.

    python scripts/evaluate.py --emb work/emb/global_train_s0.npz                    # real-only
    python scripts/evaluate.py --emb work/emb/global_train_s0.npz \
        --synth work/synth/global_train_s0/*.npz                                   # all metrics

Writes work/results/<emb name>/<real_only|synthetic name>.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opdiff import evaluate as ev  # noqa: E402
from opdiff.io import load_embeddings, load_synthetic  # noqa: E402
from opdiff.novelty import novelty  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--synth", nargs="*", default=[])
ap.add_argument("--out", default="work/results")
ap.add_argument("--clfs", nargs="+", default=["rf", "svm", "mlp"])
ap.add_argument("--skip-real-only", action="store_true")
a = ap.parse_args()

D = load_embeddings(a.emb)
seed = D["log"]["seed"]
stem = os.path.splitext(os.path.basename(a.emb))[0]
out_dir = os.path.join(a.out, stem)
os.makedirs(out_dir, exist_ok=True)
sizes = {"n_train": int(len(D["ytr"])), "n_val": int(len(D["yva"])), "n_test": int(len(D["yte"]))}


def dump(name, obj):
    with open(os.path.join(out_dir, name + ".json"), "w") as fh:
        json.dump(obj, fh, indent=1)


if not a.skip_real_only:
    t0 = time.time()
    res = {"emb": stem, "seed": seed, **sizes, "real_only": ev.real_only(D, a.clfs, seed)}
    dump("real_only", res)
    print(f"real-only ({time.time() - t0:.0f}s):",
          {c: round(v["f1"], 4) for c, v in res["real_only"].items()}, flush=True)

for path in a.synth:
    Xs, ys, meta = load_synthetic(path)
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()
    res = {"emb": stem, "seed": seed, "synthetic": name, "meta": meta, **sizes,
           "real_vs_synthetic": ev.real_vs_synthetic(D, Xs, ys, seed),
           "tstr": ev.tstr(D, Xs, ys, a.clfs, seed),
           "trts": ev.trts(D, Xs, ys, a.clfs, seed),
           "threshold_augmentation": ev.threshold_augmentation(D, Xs, ys, seed),
           "cosine": ev.cosine_similarity(D, Xs, ys, seed),
           "novelty": novelty(D, Xs, ys)}
    dump(name, res)
    ta = res["threshold_augmentation"]
    print(f"{name} ({time.time() - t0:.0f}s): TSTR-RF F1 {res['tstr']['rf']['f1']:.4f} | "
          f"aug t={ta['chosen_t']} F1 {ta['baseline_test']['f1']:.4f} -> {ta['chosen_test']['f1']:.4f} | "
          f"DCR median {res['novelty']['synthetic']['dcr_median']:.4f} "
          f"(held-out real {res['novelty']['held_out_real']['dcr_median']:.4f})", flush=True)
