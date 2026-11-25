import json

import numpy as np


def save_embeddings(path, X, split_df, families, log):
    fam_id = {f: i for i, f in enumerate(families)}
    np.savez_compressed(path, X=X.astype(np.float32),
                        y=split_df.family.map(fam_id).to_numpy(np.int64),
                        split=split_df.split.to_numpy().astype("U5"),
                        file_id=split_df.file_id.to_numpy(np.int64),
                        families=np.array(families), log=json.dumps(log))


def load_embeddings(path):
    z = np.load(path, allow_pickle=False)
    X, y, split = z["X"], z["y"], z["split"]
    D = {"families": z["families"].tolist(), "log": json.loads(str(z["log"]))}
    for tag, name in (("tr", "train"), ("va", "val"), ("te", "test")):
        m = split == name
        D["X" + tag], D["y" + tag] = X[m], y[m]
    D["X_all"], D["y_all"], D["split"] = X, y, split
    return D


def save_synthetic(path, Xs, ys, meta):
    np.savez_compressed(path, X=Xs.astype(np.float32), y=ys.astype(np.int64), meta=json.dumps(meta))


def load_synthetic(path):
    z = np.load(path, allow_pickle=False)
    return z["X"], z["y"], json.loads(str(z["meta"]))
