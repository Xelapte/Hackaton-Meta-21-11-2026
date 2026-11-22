#!/bin/bash
# v12 : recette du record (code v5 inchange) avec plus d'oubli sur les classes partielles (delta_part 12 et 14),
# puis extinction beta=7. ~2 min par run. N'ecrit que dans v12/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v12 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
for D in 12 14; do
  N=f20_p$D
  echo "[$(date +%H:%M)] $N"
  $PY unlearn.py --target delta --tau 1.0 --k_aug 0 --delta_full 20 --delta_part $D --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
done
echo "[$(date +%H:%M)] fini"
