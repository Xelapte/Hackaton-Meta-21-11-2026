#!/bin/bash
# Sweep v3 sequentiel sur 1 GPU (~2 min par run). Reutilise le code v2 (unlearning/v2/unlearn.py).
# Idee : cible "a queue lourde" = prediction de l'original sur 1-2 vues fortement modifiees, SANS tau.
set -u
BASE=/p/scratch/training2666/huang25/Epi84/unlearning
cd $BASE/v3 || exit 1
mkdir -p subs logs
PY=$BASE/.venv/bin/python
run () {  # nom, puis arguments
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY $BASE/v2/unlearn.py "$@" --out $BASE/v3/subs/$N.pt > logs/$N.log 2>&1 || echo "  -> ECHEC $N"
}
run k1_a10  --tau 1.0 --k_aug 1 --aug 1.0
run k2_a10  --tau 1.0 --k_aug 2 --aug 1.0
run k1_a12  --tau 1.0 --k_aug 1 --aug 1.25
run k4_a12  --tau 1.0 --k_aug 4 --aug 1.25
echo "[$(date +%H:%M)] fini"
