#!/bin/bash
# v10 : recette du record (v5/f20_p8) + inclinaison vers les classes ressemblantes, 3 forces (les cosinus sont petits, max 0.3 : tilt 10/20/40 = au plus +3/+6/+12 logits). ~2 min par run.
# Puis extinction beta=7 (comme le record f20_p8_b7). N'ecrit que dans v10/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v10 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
for T in 10 20 40; do
  N=f20_p8_t$T
  echo "[$(date +%H:%M)] $N"
  $PY unlearn.py --target delta --tau 1.0 --k_aug 0 --delta_full 20 --delta_part 8 --tilt $T --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
done
echo "[$(date +%H:%M)] fini"
