#!/bin/bash
# v13 : idees jamais testees sur la bonne base, UNE a la fois, a partir de la recette du record (f20_p10 + beta 7).
# 3 entrainements (~2-3 min chacun). N'ecrit que dans v13/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v13 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {  # nom, puis arguments ; puis extinction beta=7
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run f20_p10_s8   --k_aug 0 --delta_part 10 --delta_slope 8      # oubli proportionnel a la part retiree de la classe
run f20_p10_a10  --k_aug 0 --delta_part 10 --aug_train 1.0      # copies d'entrainement plus modifiees
run f20_k8_d8    --k_aug 8 --aug 1.0 --delta_part 8             # cible moyennee + oubli plus fort que v11
echo "[$(date +%H:%M)] fini"
