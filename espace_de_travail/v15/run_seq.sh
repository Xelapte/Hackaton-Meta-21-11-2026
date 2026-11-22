#!/bin/bash
# v15 : exploration de la cible moyennee (l'idee qui a donne le record v11/f20_k8_d6_b7 = 0.1462).
# Un element change a la fois autour du record (k_aug 8, aug 1.0, delta_part 6). 3 entrainements (~3 min chacun).
# N'ecrit que dans v15/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v15 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {  # nom, puis arguments ; puis extinction beta=7
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run f20_k16_d6      --k_aug 16 --aug 1.0     # plus de vues (moyenne plus lisse)
run f20_k8_a125_d6  --k_aug 8  --aug 1.25    # vues plus fortement modifiees
run f20_k8_a075_d6  --k_aug 8  --aug 0.75    # vues moins fortement modifiees
echo "[$(date +%H:%M)] fini"
