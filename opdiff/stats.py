"""Corrected resampled t-test (Nadeau & Bengio, 2003) and Holm correction."""
import numpy as np
from scipy import stats


def corrected_ttest(a, b, n_train, n_test):
    """Paired over seeds: is method a different from method b?"""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    if J < 2 or np.var(d, ddof=1) == 0:
        return {"mean_diff": float(d.mean()) if J else float("nan"), "t": float("nan"), "p": float("nan"), "J": J}
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * np.var(d, ddof=1))
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(p), "J": J}


def holm(pvals):
    """Holm-adjusted p-values, same order as the input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
