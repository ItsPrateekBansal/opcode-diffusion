"""The paper's six metrics, with accuracy, precision, recall and F1 everywhere.

1. Real vs synthetic classification (ideal F1 = 0.5)
2. t-SNE (separate script)
3. Train on synthetic, test on real (TSTR, paper Table 8)
4. Threshold augmentation (paper Table 9); t is chosen on the validation split
5. Cosine similarity, synthetic vs real, with a real vs real reference
6. Train on real, test on synthetic (TRTS, paper Table 10)
Plus novelty (distance to the closest training vector) from novelty.py.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))}


def make_clf(name, seed):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=seed)
    raise ValueError(name)


def fit(name, X, y, seed, svm_max=30000):
    if name == "svm" and len(X) > svm_max:  # stratified subsample, as in the baseline
        rng = np.random.default_rng(seed)
        keep = np.concatenate([rng.permutation(np.where(y == c)[0])[:max(1, int(svm_max * (y == c).mean()))]
                               for c in np.unique(y)])
        X, y = X[keep], y[keep]
    return make_clf(name, seed).fit(X, y)


def real_only(D, clfs, seed):
    return {c: scores(D["yte"], fit(c, D["Xtr"], D["ytr"], seed).predict(D["Xte"])) for c in clfs}


def tstr(D, Xs, ys, clfs, seed):
    return {c: scores(D["yte"], fit(c, Xs, ys, seed).predict(D["Xte"])) for c in clfs}


def trts(D, Xs, ys, clfs, seed):
    return {c: scores(ys, fit(c, D["Xtr"], D["ytr"], seed).predict(Xs)) for c in clfs}


def real_vs_synthetic(D, Xs, ys, seed, n_max=2000):
    """Per family: can a random forest tell real training vectors from synthetic?"""
    rng = np.random.default_rng(seed)
    per = {}
    for c in np.unique(ys):
        R, S = D["Xtr"][D["ytr"] == c], Xs[ys == c]
        n = min(len(R), len(S), n_max)
        if n < 10:
            continue
        R, S = R[rng.permutation(len(R))[:n]], S[rng.permutation(len(S))[:n]]
        X = np.vstack([R, S]); y = np.r_[np.zeros(n), np.ones(n)]
        perm = rng.permutation(2 * n); cut = int(0.7 * 2 * n)
        tr, te = perm[:cut], perm[cut:]
        pred = make_clf("rf", seed).fit(X[tr], y[tr]).predict(X[te])
        per[int(c)] = scores(y[te], pred)
    mean = {k: float(np.mean([v[k] for v in per.values()])) for k in next(iter(per.values()))}
    return {"mean": mean, "per_family": per}


def _per_class_f1(y, pred, classes):
    return dict(zip(classes, f1_score(y, pred, labels=classes, average=None, zero_division=0)))


def threshold_augmentation(D, Xs, ys, seed, grid=(0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99),
                           min_unique=100, paper_t=0.95):
    """Add synthetic data for families whose validation F1 is below t, or that
    have fewer than `min_unique` unique training vectors. Pick t on validation."""
    classes = sorted(np.unique(D["ytr"]).tolist())
    base = fit("rf", D["Xtr"], D["ytr"], seed)
    val_f1 = _per_class_f1(D["yva"], base.predict(D["Xva"]), classes)
    uniq = {c: len(np.unique(np.round(D["Xtr"][D["ytr"] == c], 6), axis=0)) for c in classes}
    res = {"baseline_test": scores(D["yte"], base.predict(D["Xte"])),
           "baseline_val_f1": float(f1_score(D["yva"], base.predict(D["Xva"]), average="macro")),
           "by_t": {}}
    cache = {}
    for t in grid:
        flagged = tuple(c for c in classes if val_f1[c] < t or uniq[c] < min_unique)
        if flagged not in cache:
            if flagged:
                m = np.isin(ys, flagged)
                clf = fit("rf", np.vstack([D["Xtr"], Xs[m]]), np.r_[D["ytr"], ys[m]], seed)
            else:
                clf = base
            cache[flagged] = {"val_f1": float(f1_score(D["yva"], clf.predict(D["Xva"]), average="macro")),
                              "test": scores(D["yte"], clf.predict(D["Xte"]))}
        res["by_t"][str(t)] = {"flagged": list(flagged), **cache[flagged]}
    best = max(grid, key=lambda t: res["by_t"][str(t)]["val_f1"])
    res["chosen_t"] = best
    res["chosen_test"] = res["by_t"][str(best)]["test"]
    res["paper_t_test"] = res["by_t"][str(paper_t)]["test"]
    return res


def cosine_similarity(D, Xs, ys, seed, n=500):
    """Median cosine similarity per family: synthetic vs real, and real vs real."""
    rng = np.random.default_rng(seed)
    def cos(A, B):
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return A @ B.T
    syn, ref = [], []
    for c in np.unique(ys):
        R, S = D["Xtr"][D["ytr"] == c], Xs[ys == c]
        R = R[rng.permutation(len(R))[:n]]; S = S[rng.permutation(len(S))[:n]]
        if len(R) < 2 or len(S) < 1:
            continue
        syn.append(np.median(cos(S, R)))
        rr = cos(R, R)
        ref.append(np.median(rr[~np.eye(len(R), dtype=bool)]))
    return {"synthetic_vs_real": float(np.median(syn)), "real_vs_real": float(np.median(ref))}
