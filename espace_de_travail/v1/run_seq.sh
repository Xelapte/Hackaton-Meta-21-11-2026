#!/bin/bash
# Sweep v1 SEQUENTIEL sur UN seul GPU (celui de la file qui execute la commande).
# Pas de sbatch, pas de CUDA_VISIBLE_DEVICES : a deposer dans la file du GPU 1.
# Usage : bash /p/scratch/training2666/huang25/Epi84/unlearning/v1/run_seq.sh
set -u
cd /p/scratch/training2666/huang25/Epi84/unlearning/v1 || exit 1
mkdir -p subs logs
PY=../.venv/bin/python

for T in 1.0 1.5 2.0 3.0; do
  echo "[$(date +%H:%M)] tau=$T"
  $PY unlearn.py --tau $T --epochs 3 --lr 1e-5 --out subs/tau${T}.pt > logs/tau${T}.log 2>&1 \
    || echo "  -> ECHEC tau=$T (voir logs/tau${T}.log)"
done
echo "[$(date +%H:%M)] fini"
