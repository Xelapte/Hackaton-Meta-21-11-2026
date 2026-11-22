#!/bin/bash
# v25 : forme de la cible des classes entieres, autour du record (v18/k16_wr1_at025_b7 = 0.1552). Un element a la fois.
# 4 entrainements (~4 min chacun), puis extinction beta=7. N'ecrit que dans v25/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v25 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run mfa        --w_ret 1 --aug_train 0.25 --mask_full_all
run tf05       --w_ret 1 --aug_train 0.25 --tau_full 0.5
run tf2        --w_ret 1 --aug_train 0.25 --tau_full 2.0
run wr03_at01  --w_ret 0.3 --aug_train 0.1
echo "[$(date +%H:%M)] fini"
