#!/usr/bin/env bash
# GPU stage, run inside the Kaggle notebook (or any GPU machine).
#   bash kaggle/run_gpu.sh core 0        # diffusion (paper + start sweep), GAN, SMOTE/ADASYN/noise, evaluation
#   bash kaggle/run_gpu.sh wgangp 0      # WGAN-GP (slow, about 5 h) + its evaluation
#   bash kaggle/run_gpu.sh replicate 0   # the paper's setup (per-family Word2Vec, generators see all data)
set -euo pipefail
# Smoke test first (a few minutes):  GEN_ARGS="--epochs 1 --n-per-family 50 --families 3" bash kaggle/run_gpu.sh core 0
PLAN=${1:-core}
SEED=${2:-0}

case "$PLAN" in
core)
  E=work/emb/global_train_s${SEED}.npz
  S=work/synth/global_train_s${SEED}
  python scripts/evaluate.py --emb "$E"                                     # real-only baselines
  python scripts/generate.py ${GEN_ARGS:-} --emb "$E" --method diffusion --train-tmax 500  --start-t 500
  python scripts/generate.py ${GEN_ARGS:-} --emb "$E" --method diffusion --train-tmax 1000 --start-t 250 500 750 1000
  python scripts/generate.py ${GEN_ARGS:-} --emb "$E" --method gan
  for m in smote adasyn noise; do python scripts/generate.py ${GEN_ARGS:-} --emb "$E" --method "$m"; done
  python scripts/evaluate.py --emb "$E" --skip-real-only --synth \
      "$S/diffusion_tmax500_start500.npz" "$S/gan.npz"
  python scripts/evaluate.py --emb "$E" --skip-real-only --clfs rf --synth \
      "$S"/diffusion_tmax1000_start*.npz "$S/smote.npz" "$S/adasyn.npz" "$S/noise.npz"
  python scripts/tsne.py --emb "$E" --synth "$S/diffusion_tmax500_start500.npz"
  python scripts/tsne.py --emb "$E" --synth "$S/diffusion_tmax1000_start1000.npz"
  ;;
wgangp)
  E=work/emb/global_train_s${SEED}.npz
  python scripts/generate.py ${GEN_ARGS:-} --emb "$E" --method wgangp
  python scripts/evaluate.py --emb "$E" --skip-real-only --synth work/synth/global_train_s${SEED}/wgangp.npz
  ;;
replicate)
  E=work/emb/per_family_all_s${SEED}.npz
  python scripts/evaluate.py --emb "$E"
  python scripts/generate.py ${GEN_ARGS:-} --emb "$E" --method diffusion --train-tmax 500 --start-t 500 --fit all
  python scripts/evaluate.py --emb "$E" --skip-real-only --synth \
      work/synth/per_family_all_s${SEED}/diffusion_tmax500_start500_fitall.npz
  ;;
*) echo "unknown plan: $PLAN"; exit 1 ;;
esac

python scripts/summarize.py > /dev/null
echo "done: $PLAN seed $SEED. Tables in work/summary/summary.md"
