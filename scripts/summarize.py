"""Step 5: collect all result files into tables (mean +/- sd over seeds).

    python scripts/summarize.py --results work/results --out work/summary
Writes summary.md (paste-ready tables) and rows.csv (one row per run).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opdiff.stats import corrected_ttest, holm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="work/results")
ap.add_argument("--out", default="work/summary")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = []
for path in sorted(glob.glob(os.path.join(a.results, "*", "*.json"))):
    r = json.load(open(path))
    setup = re.sub(r"_s\d+$", "", r["emb"])
    base = {"setup": setup, "seed": r["seed"], "n_train": r["n_train"], "n_test": r["n_test"]}
    if "real_only" in r:
        for clf, m in r["real_only"].items():
            rows.append({**base, "method": "real only", "metric_group": "real_only", "clf": clf, **m})
        continue
    meth = r["synthetic"]
    for grp in ("tstr", "trts"):
        for clf, m in r[grp].items():
            rows.append({**base, "method": meth, "metric_group": grp, "clf": clf, **m})
    ta = r["threshold_augmentation"]
    rows.append({**base, "method": meth, "metric_group": "aug_baseline", "clf": "rf", **ta["baseline_test"]})
    rows.append({**base, "method": meth, "metric_group": "aug_chosen_t", "clf": "rf", **ta["chosen_test"],
                 "t": ta["chosen_t"], "flagged": len(ta["by_t"][str(ta["chosen_t"])]["flagged"])})
    rows.append({**base, "method": meth, "metric_group": "aug_t0.95", "clf": "rf", **ta["paper_t_test"]})
    rows.append({**base, "method": meth, "metric_group": "real_vs_synthetic", "clf": "rf",
                 **r["real_vs_synthetic"]["mean"]})
    rows.append({**base, "method": meth, "metric_group": "cosine", "clf": "-",
                 "cos_syn_real": r["cosine"]["synthetic_vs_real"], "cos_real_real": r["cosine"]["real_vs_real"]})
    nov = r["novelty"]
    rows.append({**base, "method": meth, "metric_group": "novelty", "clf": "-",
                 **{f"syn_{k}": v for k, v in nov["synthetic"].items()},
                 **{f"real_{k}": v for k, v in nov["held_out_real"].items()}})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "rows.csv"), index=False)


def fmt(s):
    s = s.dropna()
    if not len(s):
        return ""
    return f"{s.mean():.4f}" if len(s) == 1 else f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"


def table(sub, cols, keys=("setup", "method", "clf")):
    g = sub.groupby(list(keys))
    out = g[cols].agg(fmt)
    out.insert(0, "seeds", g.size())
    return out.reset_index().to_markdown(index=False)


M = ["accuracy", "precision", "recall", "f1"]
md = ["# Results summary", "", "Values are mean ± sd over seeds (one value = one seed).", ""]
sections = [("Real-only baselines (test set)", "real_only", M),
            ("Train on synthetic, test on real (paper Table 8)", "tstr", M),
            ("Train on real, test on synthetic (paper Table 10)", "trts", M),
            ("Real vs synthetic classification, mean over families (ideal F1 = 0.5)", "real_vs_synthetic", M),
            ("Threshold augmentation: RF before (baseline)", "aug_baseline", M),
            ("Threshold augmentation: RF after, t chosen on validation", "aug_chosen_t", M + ["t", "flagged"]),
            ("Threshold augmentation: RF after, t = 0.95 (paper)", "aug_t0.95", M),
            ("Cosine similarity (median)", "cosine", ["cos_syn_real", "cos_real_real"]),
            ("Novelty: distance to closest training vector", "novelty",
             ["syn_dcr_median", "syn_dcr_p5", "syn_memorized_share", "syn_copy_share",
              "real_dcr_median", "real_memorized_share"])]
for title, grp, cols in sections:
    sub = df[df.metric_group == grp]
    if len(sub):
        md += [f"## {title}", "", table(sub, [c for c in cols if c in sub]), ""]

# significance: augmentation gain vs no augmentation, paired over seeds
tests = []
for (setup, meth), g in df[df.metric_group.isin(["aug_baseline", "aug_chosen_t"])].groupby(["setup", "method"]):
    piv = g.pivot_table(index="seed", columns="metric_group", values="f1")
    if {"aug_baseline", "aug_chosen_t"} <= set(piv.columns) and len(piv) >= 2:
        res = corrected_ttest(piv.aug_chosen_t, piv.aug_baseline, g.n_train.iloc[0], g.n_test.iloc[0])
        tests.append({"setup": setup, "method": meth, **res})
if tests:
    T = pd.DataFrame(tests)
    ok = T.p.notna()
    T.loc[ok, "p_holm"] = holm(T.loc[ok, "p"].tolist())
    md += ["## Augmentation gain in macro-F1 vs no augmentation (corrected resampled t-test, Holm)", "",
           T.to_markdown(index=False, floatfmt=".4f"), ""]

open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md))
print("\n".join(md))
