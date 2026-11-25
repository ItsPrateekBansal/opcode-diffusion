"""Reading the opcode corpus, finding duplicates, and splitting.

The corpus is a zip (or folder) where each file holds the opcode mnemonics of
one malware sample and the parent folder name is the family label.

The full corpus is about 95 GB unzipped, so files are read in chunks and only
two things are kept per file:
  - the count of every opcode (enough for the mean-pooled vector, exactly), and
  - the first `keep_first` opcodes (what Word2Vec learns from; gensim's own
    default also stops at 10,000 words per sentence).
"""
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import sparse

UNKNOWN_NAMES = {"unknown_family", "unknown"}
CHUNK_BYTES = 8 << 20


# ---------------------------------------------------------------- reading
def _is_junk(path):
    parts = path.replace("\\", "/").split("/")
    return any(p.startswith(".") or p == "__MACOSX" for p in parts)


def family_of(path):
    """Family label = name of the folder that directly holds the file."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    return parts[-2] if len(parts) >= 2 else None


class Corpus:
    """Uniform access to a zip file or an extracted folder."""

    def __init__(self, src):
        self.src = src
        self.is_zip = os.path.isfile(src) and zipfile.is_zipfile(src)
        self._zf = zipfile.ZipFile(src) if self.is_zip else None

    def paths(self):
        if self.is_zip:
            for info in self._zf.infolist():
                if not info.is_dir() and not _is_junk(info.filename):
                    yield info.filename
        else:
            for root, _, files in os.walk(self.src):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), self.src)
                    if not _is_junk(rel):
                        yield rel

    def open(self, path):
        return self._zf.open(path) if self.is_zip else open(os.path.join(self.src, path), "rb")

    def read(self, path, limit=None):
        with self.open(path) as fh:
            return fh.read(limit) if limit else fh.read()


def _tokens_from_chunk(buf, mode):
    if mode == "first":
        return [ln.split(None, 1)[0] for ln in buf.splitlines() if ln.strip()]
    return buf.split()


def scan_file(corpus, path, mode="whitespace", token_regex=None, keep_first=10000):
    """Stream one file. Returns (n_tokens, sha1, Counter, first tokens)."""
    pat = re.compile(token_regex.encode()) if token_regex else None
    counts, first, n = Counter(), [], 0
    h = hashlib.sha1()
    rest = b""
    with corpus.open(path) as fh:
        while True:
            data = fh.read(CHUNK_BYTES)
            buf = rest + data.lower()
            if data:
                cut = buf.rfind(b"\n") if mode == "first" else max(buf.rfind(b"\n"), buf.rfind(b" "))
                if cut < 0:
                    rest = buf
                    continue
                buf, rest = buf[:cut + 1], buf[cut + 1:]
            toks = _tokens_from_chunk(buf, mode)
            if pat:
                toks = [t for t in toks if pat.fullmatch(t)]
            if toks:
                counts.update(toks)
                h.update(b" ".join(toks) + b" ")
                if len(first) < keep_first:
                    first.extend(toks[: keep_first - len(first)])
                n += len(toks)
            if not data:
                break
    return n, h.hexdigest(), counts, first


def tokenize(raw, mode="whitespace", token_regex=None):
    """Used by inspect_data.py on small samples."""
    toks = _tokens_from_chunk(raw.lower(), mode)
    if token_regex:
        pat = re.compile(token_regex.encode())
        toks = [t for t in toks if pat.fullmatch(t)]
    return [t.decode("utf-8", "ignore") for t in toks]


_W = {}


def _init_worker(src, mode, token_regex, keep_first):
    _W.update(corpus=Corpus(src), mode=mode, regex=token_regex, keep=keep_first)


def _scan_worker(path):
    n, sha1, counts, first = scan_file(_W["corpus"], path, _W["mode"], _W["regex"], _W["keep"])
    return path, n, sha1, counts, b" ".join(first)


# ---------------------------------------------------------------- token store
class TokenStore:
    """Per-file opcode counts (sparse) plus the first opcodes of each file."""

    def __init__(self, index_dir):
        self.dir = index_dir
        self.index = pd.read_csv(os.path.join(index_dir, "index.csv"))
        with open(os.path.join(index_dir, "vocab.json")) as fh:
            self.vocab = json.load(fh)  # list: id -> opcode
        self.vocab_arr = np.array(self.vocab, dtype=object)
        path = os.path.join(index_dir, "first_tokens.u16")
        self.tokens = (np.memmap(path, dtype=np.uint16, mode="r") if os.path.getsize(path)
                       else np.zeros(0, dtype=np.uint16))
        self.counts = sparse.load_npz(os.path.join(index_dir, "counts.npz")).tocsr()
        self._row = dict(zip(self.index.file_id, range(len(self.index))))
        self._off = dict(zip(self.index.file_id, self.index.first_offset))
        self._len = dict(zip(self.index.file_id, self.index.first_len))

    def ids(self, file_id):
        o, n = self._off[file_id], self._len[file_id]
        return np.asarray(self.tokens[o:o + n])

    def words(self, file_id):
        return self.vocab_arr[self.ids(file_id)].tolist()

    def count_rows(self, file_ids):
        return self.counts[[self._row[f] for f in file_ids]]


def build_index(src, out_dir, top_k=25, mode="whitespace", token_regex=None,
                keep_first=10000, workers=3, log_every=2000):
    """Pick the top_k largest named families and record counts, hashes and first opcodes."""
    os.makedirs(out_dir, exist_ok=True)
    corpus = Corpus(src)
    all_paths = list(corpus.paths())
    fam_counts = Counter(family_of(p) for p in all_paths)
    fam_counts.pop(None, None)
    unknown = {f: c for f, c in fam_counts.items() if f.lower() in UNKNOWN_NAMES}
    named = {f: c for f, c in fam_counts.items() if f.lower() not in UNKNOWN_NAMES}
    top = [f for f, _ in sorted(named.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
    top_set = set(top)
    selected = [p for p in all_paths if family_of(p) in top_set]
    print(f"{len(all_paths):,} files, {len(fam_counts):,} folders; "
          f"reading {len(selected):,} files from the top {top_k} families", flush=True)

    vocab, vocab_id = [], {}
    rows, crow, ccol, cval = [], [], [], []
    offset = 0
    args = (src, mode, token_regex, keep_first)
    with open(os.path.join(out_dir, "first_tokens.u16"), "wb") as out, \
            Pool(workers, initializer=_init_worker, initargs=args) as pool:
        for i, (path, n, sha1, counts, first) in enumerate(
                pool.imap(_scan_worker, selected, chunksize=16)):
            for tok, c in counts.items():
                j = vocab_id.get(tok)
                if j is None:
                    j = vocab_id[tok] = len(vocab)
                    vocab.append(tok.decode("utf-8", "ignore"))
                crow.append(i); ccol.append(j); cval.append(c)
            if len(vocab) > 65535:
                raise RuntimeError("More than 65,535 distinct tokens: the files probably hold full "
                                   "instructions. Run inspect_data.py and try --tokenize first.")
            ids = np.fromiter((vocab_id[t] for t in first.split()), dtype=np.uint16) if first else \
                np.zeros(0, dtype=np.uint16)
            out.write(ids.tobytes())
            rows.append((i, path, family_of(path), n, sha1, offset, len(ids)))
            offset += len(ids)
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1:,}/{len(selected):,} files, {len(vocab):,} distinct opcodes", flush=True)

    index = pd.DataFrame(rows, columns=["file_id", "path", "family", "n_tokens", "sha1",
                                        "first_offset", "first_len"])
    index.to_csv(os.path.join(out_dir, "index.csv"), index=False)
    C = sparse.csr_matrix((np.asarray(cval, dtype=np.float64), (crow, ccol)),
                          shape=(len(index), len(vocab)))
    sparse.save_npz(os.path.join(out_dir, "counts.npz"), C)
    with open(os.path.join(out_dir, "vocab.json"), "w") as fh:
        json.dump(vocab, fh)

    summary = {
        "source": os.path.basename(src),
        "files_total": len(all_paths),
        "family_folders": len(fam_counts),
        "unknown_family_files": int(sum(unknown.values())),
        "top_families": top,
        "files_selected": int(len(index)),
        "empty_files": int((index.n_tokens == 0).sum()),
        "vocab_size": len(vocab),
        "tokens_total": int(index.n_tokens.sum()),
        "tokens_median_per_file": float(index.n_tokens.median()),
        "keep_first": keep_first,
        "duplicates": duplicate_report(index),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def duplicate_report(index):
    """Exact duplicates by content hash, per family and across families."""
    nonempty = index[index.n_tokens > 0]
    per_fam = (nonempty.groupby("family")
               .agg(files=("sha1", "size"), unique=("sha1", "nunique"))
               .sort_values("files", ascending=False))
    fams_per_hash = nonempty.groupby("sha1").family.nunique()
    files_per_hash = nonempty.groupby("sha1").size()
    return {
        "per_family": {f: {"files": int(r["files"]), "unique": int(r["unique"])}
                       for f, r in per_fam.iterrows()},
        "unique_contents": int(nonempty.sha1.nunique()),
        "hashes_in_more_than_one_family": int((fams_per_hash > 1).sum()),
        "max_copies_of_one_file": int(files_per_hash.max()) if len(files_per_hash) else 0,
    }


# ---------------------------------------------------------------- split
def make_split(index, seed, val_frac=0.1, test_frac=0.2):
    """Split by unique content so no copy of a test file is in training.

    - Files with identical opcodes share a hash; a hash goes to one side only.
    - Hashes seen in more than one family are ambiguous and go to train only.
    - Train keeps every file (duplicates included, as in the paper).
    - Validation and test keep one file per hash.
    """
    df = index[index.n_tokens > 0].copy()
    fams_per_hash = df.groupby("sha1").family.nunique()
    multi = set(fams_per_hash[fams_per_hash > 1].index)
    rng = np.random.default_rng(seed)
    side = {h: "train" for h in multi}
    for fam, g in df[~df.sha1.isin(multi)].groupby("family"):
        hashes = np.array(sorted(g.sha1.unique()))
        rng.shuffle(hashes)
        n = len(hashes)
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        for h in hashes[:n_test]:
            side[h] = "test"
        for h in hashes[n_test:n_test + n_val]:
            side[h] = "val"
        for h in hashes[n_test + n_val:]:
            side[h] = "train"
    df["split"] = df.sha1.map(side)
    df = df[(df.split == "train") | ~df.duplicated("sha1")]
    return df[["file_id", "family", "sha1", "split"]].reset_index(drop=True)
