#!/bin/bash
# v17 : plus de vues dans la cible moyennee (8 vues = 0.1462, 16 vues = 0.1506 -> 32 et 64), et copies d'entrainement
# plus legeres (copies plus fortes = 0.1307 < 0.1412). Recette du record sinon. Puis extinction beta=7.
# ~5 + 4 + 9 min. N'ecrit que dans v17/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v17 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {  # nom, puis arguments ; puis extinction beta=7
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run f20_k32_d6       --k_aug 32
run f20_k16_d6_at025 --k_aug 16 --aug_train 0.25
run f20_k64_d6       --k_aug 64
echo "[$(date +%H:%M)] fini"
