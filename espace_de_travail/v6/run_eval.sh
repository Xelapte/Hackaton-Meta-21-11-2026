#!/bin/bash
# v6 : evaluation seule (pas d'entrainement), ~2-3 min sur 1 GPU. A deposer dans la file du GPU 1.
# Mesure l'effet de la baisse du biais des 9 classes entieres (beta) sur les meilleurs modeles existants.
set -u
BASE=/p/scratch/training2666/huang25/Epi84/unlearning
cd $BASE/v6 || exit 1
mkdir -p logs
$BASE/.venv/bin/python eval.py \
  $BASE/model.pt \
  $BASE/v4/subs/d12.pt $BASE/v4/subs/d16.pt \
  $BASE/v5/subs/f20_p8.pt $BASE/v5/subs/f20_p8_wr1.pt \
  --betas 0 4 8 12 16 > logs/eval.log 2>&1 || echo "ECHEC eval (voir logs/eval.log)"
echo "fini"
