#!/bin/bash
# v11 : recette du record (v5/f20_p8, sans professeur v9) + cible moyennee sur 8 vues modifiees, 2 forces d'oubli.
# Puis extinction beta=7 (comme le record f20_p8_b7). ~3 min par run. N'ecrit que dans v11/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v11 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
for D in 6 4; do
  N=f20_k8_d$D
  echo "[$(date +%H:%M)] $N"
  $PY unlearn.py --target delta --tau 1.0 --k_aug 8 --aug 1.0 --delta_full 20 --delta_part $D --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
done
echo "[$(date +%H:%M)] fini"
