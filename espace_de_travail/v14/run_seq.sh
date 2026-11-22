#!/bin/bash
# v14 : recette du record (v11/f20_k8_d6 + beta 7) avec masque de saillance SalUn : seule une fraction des poids bouge.
# 2 entrainements (~3 min chacun). N'ecrit que dans v14/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v14 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
for F in 0.5 0.2; do
  N=f20_k8_d6_sal$F
  echo "[$(date +%H:%M)] $N"
  $PY unlearn.py --target delta --tau 1.0 --k_aug 8 --aug 1.0 --delta_full 20 --delta_part 6 --salun_frac $F --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
done
echo "[$(date +%H:%M)] fini"
