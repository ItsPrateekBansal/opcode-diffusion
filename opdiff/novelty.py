"""How close synthetic vectors are to the training data.

DCR    distance to the closest training vector of the same family,
       Euclidean divided by sqrt(dim) (as used in TabDDPM).
NNDR   d(nearest) / d(second nearest). A sample counts as memorized when the
       ratio is below 1/3 (Gu et al. 2025; Fang et al. 2025).
Copy   DCR below 1e-6.
The same numbers for held-out real vectors (validation + test) give the
reference: what "new but realistic" looks like.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _dist(Q, R):
    k = min(2, len(R))
    d, _ = NearestNeighbors(n_neighbors=k).fit(R).kneighbors(Q)
    d = d / np.sqrt(R.shape[1])
    d1 = d[:, 0]
    ratio = d1 / d[:, 1] if k == 2 else np.zeros_like(d1)
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    return d1, ratio


def _summary(d1, ratio):
    return {"dcr_median": float(np.median(d1)), "dcr_p5": float(np.percentile(d1, 5)),
            "memorized_share": float((ratio < 1 / 3).mean()),
            "copy_share": float((d1 < 1e-6).mean()), "n": int(len(d1))}


def novelty(D, Xs, ys):
    syn_d, syn_r, ref_d, ref_r = [], [], [], []
    for c in np.unique(ys):
        R = np.unique(np.round(D["Xtr"][D["ytr"] == c], 6), axis=0)
        if len(R) == 0:
            continue
        d, r = _dist(Xs[ys == c], R); syn_d.append(d); syn_r.append(r)
        H = np.vstack([D["Xva"][D["yva"] == c], D["Xte"][D["yte"] == c]])
        if len(H):
            d, r = _dist(H, R); ref_d.append(d); ref_r.append(r)
    return {"synthetic": _summary(np.concatenate(syn_d), np.concatenate(syn_r)),
            "held_out_real": _summary(np.concatenate(ref_d), np.concatenate(ref_r))}
