"""Simple augmentation baselines that make n synthetic vectors per family.

SMOTE and ADASYN use imbalanced-learn and run on the unique training vectors
(running them on exact duplicates would mostly produce copies). Gaussian noise
adds N(0, (sigma * feature_std)^2) to random training vectors.
"""
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE


def _unique(X, y):
    _, idx = np.unique(np.round(X, 6), axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx], y[idx]


def oversample(X, y, n_per_class, method="smote", seed=0, k=5):
    """Return (Xs, ys, log). One class at a time so the budget is exact."""
    Xu, yu = _unique(X, y)
    out_X, out_y, log = [], [], {}
    for c in np.unique(y):
        n_c = int((yu == c).sum())
        kk = min(k, n_c - 1)
        if kk < 1:  # one unique vector: nothing to interpolate
            log[int(c)] = "only one unique vector: copies"
            out_X.append(np.repeat(Xu[yu == c], n_per_class, axis=0))
            out_y.append(np.full(n_per_class, c))
            continue
        strategy = {c: n_c + n_per_class}
        try:
            if method == "smote":
                sm = SMOTE(sampling_strategy=strategy, k_neighbors=kk, random_state=seed)
            else:
                sm = ADASYN(sampling_strategy=strategy, n_neighbors=kk, random_state=seed)
            Xr, yr = sm.fit_resample(Xu, yu)
            new = Xr[len(Xu):]
            new = new[yr[len(Xu):] == c]
        except (ValueError, RuntimeError) as e:
            # ADASYN refuses when a class has no hard (mixed) neighbourhoods
            log[int(c)] = f"no samples: {e}"
            new = np.zeros((0, X.shape[1]), dtype=np.float32)
        out_X.append(new.astype(np.float32))
        out_y.append(np.full(len(new), c))
    return np.concatenate(out_X), np.concatenate(out_y), log


def gaussian_noise(X, y, n_per_class, sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    std = X.std(axis=0)
    out_X, out_y = [], []
    for c in np.unique(y):
        base = X[y == c][rng.integers(0, (y == c).sum(), n_per_class)]
        noisy = base + rng.normal(0, 1, base.shape) * sigma * std
        out_X.append(np.clip(noisy, -1, 1).astype(np.float32))
        out_y.append(np.full(n_per_class, c))
    return np.concatenate(out_X), np.concatenate(out_y), {"sigma": sigma}
