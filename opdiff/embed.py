"""Word2Vec embeddings, mean-pooled to one vector per file, scaled to [-1, 1].

Two modes:
  per_family  one Word2Vec model per family, as in Bao et al. (2025).
              The model used to embed a file is chosen by the file's label.
  global      one Word2Vec model for all families.

`fit` decides which files the Word2Vec models and the scaler learn from:
  all    every file (the paper's setup)
  train  training files only (no test information leaks into the features)
"""
import os
import tempfile

import numpy as np
from gensim.models import Word2Vec

CHUNK = 10000  # gensim silently truncates longer sentences, so we chunk


def _write_corpus(store, file_ids, path, max_tokens=0):
    """One line per chunk of each file's first opcodes (see data.py)."""
    with open(path, "w") as fh:
        for fid in file_ids:
            words = store.words(fid)
            if max_tokens:
                words = words[:max_tokens]
            for s in range(0, len(words), CHUNK):
                fh.write(" ".join(words[s:s + CHUNK]) + "\n")


def train_word2vec(store, file_ids, dim=104, window=5, epochs=5, sg=0,
                   min_count=1, workers=4, seed=0, max_tokens=0):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "corpus.txt")
        _write_corpus(store, file_ids, path, max_tokens)
        return Word2Vec(corpus_file=path, vector_size=dim, window=window,
                        epochs=epochs, sg=sg, min_count=min_count,
                        workers=workers, seed=seed)


def _lookup_matrix(model, vocab):
    """Rows indexed by our token ids; missing tokens get has=False."""
    dim = model.vector_size
    mat = np.zeros((len(vocab), dim), dtype=np.float32)
    has = np.zeros(len(vocab), dtype=bool)
    for i, tok in enumerate(vocab):
        if tok in model.wv.key_to_index:
            mat[i] = model.wv[tok]
            has[i] = True
    return mat, has


def mean_pool(store, file_ids, mat, has):
    """Mean over every opcode of each file, from the stored counts (exact)."""
    C = store.count_rows(file_ids)[:, np.where(has)[0]]
    total = np.asarray(C.sum(axis=1)).ravel()
    out = np.asarray(C @ mat[has], dtype=np.float32)
    ok = total > 0
    out[ok] /= total[ok, None]
    return out.astype(np.float32), ok


def build_embeddings(store, split, mode="global", fit="train", seed=0, **w2v):
    """Return X (n x dim, in [-1, 1]), keep-mask, and a log dict."""
    fids = split.file_id.to_numpy()
    fams = split.family.to_numpy()
    is_fit = np.ones(len(split), bool) if fit == "all" else (split.split == "train").to_numpy()
    X = np.zeros((len(split), w2v.get("dim", 104)), dtype=np.float32)
    ok = np.zeros(len(split), dtype=bool)
    log = {"mode": mode, "fit": fit, "w2v": w2v, "models": []}

    if mode == "global":
        model = train_word2vec(store, fids[is_fit], seed=seed, **w2v)
        mat, has = _lookup_matrix(model, store.vocab)
        X, ok = mean_pool(store, fids, mat, has)
        log["models"].append({"name": "global", "vocab": int(has.sum())})
    elif mode == "per_family":
        for fam in sorted(set(fams)):
            rows = np.where(fams == fam)[0]
            model = train_word2vec(store, fids[rows[is_fit[rows]]], seed=seed, **w2v)
            mat, has = _lookup_matrix(model, store.vocab)
            X[rows], ok[rows] = mean_pool(store, fids[rows], mat, has)
            log["models"].append({"name": fam, "vocab": int(has.sum())})
    else:
        raise ValueError(mode)

    # min-max scale every feature to [-1, 1]; fit on the same rows as Word2Vec
    ref = X[is_fit & ok]
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    X = np.clip(2 * (X - lo) / span - 1, -1, 1).astype(np.float32)

    # drop val/test rows whose vector equals a training vector
    is_train = (split.split == "train").to_numpy()
    train_keys = {r.tobytes() for r in np.round(X[is_train & ok], 6)}
    clash = np.array([(not is_train[i]) and ok[i] and
                      np.round(X[i], 6).tobytes() in train_keys for i in range(len(X))])
    log["dropped_no_known_tokens"] = int((~ok).sum())
    log["dropped_eval_equal_to_train"] = int(clash.sum())
    return X, ok & ~clash, log
