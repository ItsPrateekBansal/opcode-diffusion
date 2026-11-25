"""Small fake corpus with the same layout as the real one, for testing only.

    python scripts/make_toy_data.py --out work/toy_corpus.zip
"""
import argparse
import os
import random
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="work/toy_corpus.zip")
ap.add_argument("--families", type=int, default=6)
a = ap.parse_args()

OPS = ["mov", "push", "pop", "call", "jmp", "je", "jne", "add", "sub", "xor", "cmp", "test",
       "lea", "ret", "nop", "inc", "dec", "and", "or", "shl", "shr", "imul", "movzx", "leave"]
rnd = random.Random(0)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in range(a.families):
        weights = [rnd.random() ** 3 for _ in OPS]
        n_files = rnd.randint(60, 250)
        for i in range(n_files):
            n = rnd.randint(200, 2000)
            text = "\n".join(rnd.choices(OPS, weights, k=n))
            z.writestr(f"corpus/Fam{f}/{f}_{i}.txt", text)
            if i % 10 == 0:  # exact duplicate in the same family
                z.writestr(f"corpus/Fam{f}/{f}_{i}_copy.txt", text)
        if f == 0:  # a tiny family made of a few repeated files
            for j in range(40):
                z.writestr(f"corpus/Fam{f}/dup_{j}.txt", "\n".join(OPS[(j % 3):(j % 3) + 12] * 20))
    for j in range(20):  # "Tiny": 20 files, only 4 unique contents
        z.writestr(f"corpus/Tiny/t_{j}.txt", "\n".join(OPS[(j % 4) * 3:(j % 4) * 3 + 8] * 30))
    for i in range(80):
        z.writestr(f"corpus/Unknown_Family/u_{i}.txt", "\n".join(rnd.choices(OPS, k=300)))
    z.writestr("corpus/Fam1/empty.txt", "")
    z.writestr("__MACOSX/corpus/._junk", "x")
print("->", a.out)
