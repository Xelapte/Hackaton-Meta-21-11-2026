#!/bin/bash
# Sweep v2 sequentiel sur 1 GPU (~2 min par run). A deposer dans la file du GPU 1.
set -u
cd /p/scratch/training2666/huang25/Epi84/unlearning/v2 || exit 1
mkdir -p subs logs
PY=../.venv/bin/python
run () {  # nom, puis arguments
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py "$@" --out subs/$N.pt > logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run t2      --tau 2.0
run t3      --tau 3.0
run t4      --tau 4.0
run t3_lr1e4 --tau 3.0 --lr 1e-4
echo "[$(date +%H:%M)] fini"
