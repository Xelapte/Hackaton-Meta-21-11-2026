#!/bin/bash
# Sweep v4 sequentiel sur 1 GPU (~1 min de diag + ~2-3 min par run). A deposer dans la file du GPU 1.
# Cible delta : logits de l'original, logit de la classe memorisee baisse de delta.
set -u
BASE=/p/scratch/training2666/huang25/Epi84/unlearning
cd $BASE/v4 || exit 1
mkdir -p subs logs
PY=$BASE/.venv/bin/python
run () {  # nom, puis arguments
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py "$@" --out subs/$N.pt > logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
echo "[$(date +%H:%M)] diag"
$PY unlearn.py --diag > logs/diag.log 2>&1 || echo "  -> ECHEC diag"
# balayage large de la force d'oubli (tau=1 : pas d'aplatissement, k_aug=0 : base = image exacte)
run d4  --target delta --delta 4  --tau 1.0 --k_aug 0
run d8  --target delta --delta 8  --tau 1.0 --k_aug 0
run d12 --target delta --delta 12 --tau 1.0 --k_aug 0
run d16 --target delta --delta 16 --tau 1.0 --k_aug 0
echo "[$(date +%H:%M)] fini"
