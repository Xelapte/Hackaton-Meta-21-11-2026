#!/bin/bash
# v33 : se tromper franchement sur les ~20 % d'images les plus DIFFICILES (au lieu de classes entieres), sans extinction
# globale. 3 entrainements (~6 min chacun, 8 epoques). N'ecrit que dans v33/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v33 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --tau 1.0 --aug 1.0 --k_aug 16 --w_ret 1 --aug_train 0.25 --mos_exclude_full --epochs 8 "$@" --out subs/$N.pt > logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run h20_w4   --hard_frac 0.20 --delta_full 25 --delta_part 3 --hard_w 4
run h20_w10  --hard_frac 0.20 --delta_full 25 --delta_part 3 --hard_w 10
run h20_w10b --hard_frac 0.20 --delta_full 25 --delta_part 6 --hard_w 10 --lr 1e-4
echo "[$(date +%H:%M)] fini"
