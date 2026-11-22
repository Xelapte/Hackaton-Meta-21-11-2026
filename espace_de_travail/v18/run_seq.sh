#!/bin/bash
# v18 : autour du record (v15 : 16 vues, delta_part 6, w_ret 3, 5 epoques = 0.1506), un element a la fois :
# ancrage plus faible, entrainement plus court, puis combinaison avec des copies legeres. Puis extinction beta=7.
# ~4 min par run. N'ecrit que dans v18/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v18 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run k16_wr1        --w_ret 1
run k16_ep3        --epochs 3
run k16_wr1_at025  --w_ret 1 --aug_train 0.25
echo "[$(date +%H:%M)] fini"
