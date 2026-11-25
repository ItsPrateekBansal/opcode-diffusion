# Opcode Diffusion: Synthetic Malware Embeddings Under a Fair Test

CS 297 project, San José State University. Prateek Bansal. Advisor: Dr. Fabio Di Troia.

This repository reproduces Bao et al., "Generating Synthetic Malware Samples Using Generative AI"
(IEEE Access, 2025) and runs the first of my proposed experiments:

- **E1, leak-free embedding.** One Word2Vec model for all families, trained on training files only.
  The paper trains one Word2Vec model per family, which lets a classifier recognise the family from
  the embedding space instead of the opcodes.
- **E2, clean protocol.** The data is split by file content first. Generators see training data only.
  The augmentation threshold is chosen on a validation split. Every setting can run with several seeds.
- **E3, diffusion start point.** The paper starts diffusion sampling from real samples noised to T/2.
  Here the start point is a setting (T/4, T/2, 3T/4, T), and every synthetic set is checked for how
  close it sits to the training data.
- **E4, simple baselines.** SMOTE, ADASYN and Gaussian noise get the same budget as the generators.

All metrics report accuracy, macro precision, macro recall and macro F1.

## Data

The opcode corpus comes from VirusShare samples (https://virusshare.com). It was disassembled and
provided by my advisor as `VirusShare_opcodes_extracted_v077.zip` (17.4 GB). It is not included in
this repository. The code expects one file per sample, with the family as the folder name.

## How to run

The work is split in two. The laptop reads the 17 GB corpus and makes small embedding files.
The GPU machine (Kaggle) trains the generators and runs the evaluation.

### 1. Laptop: set up (once)

Use Python 3.10 to 3.13.

```bash
git clone https://github.com/<your-user>/opcode-diffusion.git
cd opcode-diffusion
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy pandas gensim scikit-learn
```

### 2. Laptop: check the file format (1 minute)

```bash
python scripts/inspect_data.py --src /path/to/VirusShare_opcodes_extracted_v077.zip
```

Check that the folder names are the families and the top tokens are opcode names (mov, push, call).
If you also see registers or hex values, the files hold full instructions: add `--tokenize first`
to the next command.

### 3. Laptop: index the corpus (about 15 to 45 minutes)

```bash
python scripts/index_data.py --src /path/to/VirusShare_opcodes_extracted_v077.zip --out work/index
```

This reads the zip directly (no need to extract it), keeps the 25 largest named families and drops
`Unknown_Family`. Files are read in chunks. For each file it keeps the count of every opcode (enough
for the exact mean-pooled vector) and the first 10,000 opcodes, which Word2Vec learns from
(gensim's own default limit per sentence). It also prints how many exact duplicates
each family has, and writes everything to `work/index/summary.json`.

### 4. Laptop: build the embeddings

```bash
# leak-free setup (E1 + E2): global Word2Vec, fitted on training files only, three seeds
python scripts/embed.py --mode global --fit train --seeds 0 1 2
# the paper's setup: one Word2Vec per family, fitted on all files
python scripts/embed.py --mode per_family --fit all --seeds 0
```

Each run writes one file per seed to `work/emb/` (about 30 MB each). If Word2Vec is too slow,
add `--max-tokens 5000` to learn from fewer opcodes per file.

### 5. Kaggle: upload the embeddings

Zip the `work/emb` folder and upload it as a new private Kaggle dataset named `opdiff-emb`.

### 6. Kaggle: run the GPU stage

Create a notebook with GPU (T4 or P100), Internet on, and the `opdiff-emb` dataset added. Run:

```bash
!git clone https://github.com/<your-user>/opcode-diffusion.git
%cd opcode-diffusion
!mkdir -p work/emb && find /kaggle/input -name "*.npz" -exec cp {} work/emb/ \;
# smoke test, a few minutes. Then delete the test output: !rm -rf work/synth work/results work/summary
!GEN_ARGS="--epochs 1 --n-per-family 50 --families 3" bash kaggle/run_gpu.sh core 0
# real runs
!bash kaggle/run_gpu.sh core 0         # rough guide: 3 to 5 h
!bash kaggle/run_gpu.sh replicate 0    # rough guide: 1 to 2 h
!bash kaggle/run_gpu.sh wgangp 0       # rough guide: 5 h, optional
!cd work && zip -r /kaggle/working/results.zip results summary figures
```

Use "Save Version > Save & Run All" so the run continues after you close the browser
(Kaggle sessions stop after 12 hours; split the plans across versions if needed).
Download `results.zip` from the notebook's Output tab.

## What each command produces

| Folder | Contents |
|---|---|
| `work/index/` | cached opcodes, `index.csv`, duplicate report in `summary.json` |
| `work/emb/` | `{setup}_s{seed}.npz`: vectors, labels, split |
| `work/synth/{setup}_s{seed}/` | one `.npz` per synthetic set |
| `work/results/{setup}_s{seed}/` | one `.json` per synthetic set, plus `real_only.json` |
| `work/summary/summary.md` | all tables, mean ± sd over seeds, significance tests |
| `work/figures/` | t-SNE plots |

## Settings

| Item | Value | Source |
|---|---|---|
| Families | 25 largest named, `Unknown_Family` dropped | paper |
| Word2Vec | 104 dims, window 5, 5 epochs, CBOW, min count 1 | paper (dims); my choice (rest) |
| Scaling | min-max to [-1, 1], fitted on the same rows as Word2Vec | paper |
| Split | by content hash, 70 / 10 / 20 train / val / test; duplicates stay in train | my choice |
| GAN | dense 200-400-800, BatchNorm, Tanh; Adam 3e-4, betas (0.5, 0.9) | paper Table 5 |
| WGAN-GP | Conv1D 64-32-16 generator, 64-128-256 critic, k = 3; Adam 1e-4; 5 critic steps, λ = 10 | paper Table 5; my choice (last two) |
| Diffusion | 1D U-Net, base 32, mults (1,2,4,8), attention, dropout 0.01, AdamW 5e-5; T = 1000 linear | paper Table 6 |
| Training | latent 100, batch 64 (GAN, WGAN-GP) / 128 (diffusion), 200 epochs per family | my choice |
| Synthetic | 2,000 vectors per family | my choice |
| Classifiers | RF (200 trees), SVM (RBF, 30k subsample), MLP (256-128) | paper (types) |
| Threshold | t from 0.90 to 0.99, chosen on validation; t = 0.95 also reported | paper (range) |

Two implementation notes. Keras BatchNorm momentum 0.8 is PyTorch momentum 0.2. The U-Net pads the
104 features to 128 so the length can be halved four times, and crops back to 104.

## Layout

```
opdiff/      data.py (index, split), embed.py (Word2Vec), models.py (GAN, WGAN-GP, diffusion),
             baselines.py, evaluate.py (six metrics), novelty.py, stats.py, io.py
scripts/     inspect_data, index_data, embed, generate, evaluate, summarize, tsne, make_toy_data
kaggle/      run_gpu.sh
```

To test everything on a small fake corpus: `python scripts/make_toy_data.py`, then run steps 3 to 6
with `--src work/toy_corpus.zip` and the smoke-test settings.
