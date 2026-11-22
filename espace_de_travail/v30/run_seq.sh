#!/bin/bash
# v30 : recette du record, mais le groupe B = {6,7,8} est traite comme des classes PARTIELLES (oubli modere, pas
# d'extinction). Classes entieres = {0,29,30,31,33,38}. Puis extinction beta=7 sur ces 6 classes seulement.
# 2 entrainements (~4 min chacun). N'ecrit que dans v30/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v30 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
FC="0,29,30,31,33,38"
run () {
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 --aug 1.0 --k_aug 16 --w_ret 1 --aug_train 0.25 --full_classes $FC "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt $FC >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run Bpart_d6 --delta_part 6
run Bpart_d3 --delta_part 3
echo "[$(date +%H:%M)] fini"
