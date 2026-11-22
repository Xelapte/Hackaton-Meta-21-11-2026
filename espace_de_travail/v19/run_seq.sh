#!/bin/bash
# v19 : cible "de reference par classe" pour les images des 9 classes entieres (melange image / classe).
# 4 entrainements (~4 min chacun). Chaque modele est garde SANS extinction globale (.pt) et AVEC beta=7 (_b7.pt).
# N'ecrit que dans v19/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v19 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target ref --tau 1.0 --delta_part 6 --aug 1.0 --k_aug 16 "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
run ref_l05         --ref_lambda 0.5                                  # melange image/classe, ancrage standard
run ref_l00         --ref_lambda 0.0                                  # reference de classe pure
run ref_l05_mx      --ref_lambda 0.5 --mos_exclude_full               # + mosaiques sans les classes entieres
run ref_all_l05_mx  --ref_lambda 0.5 --mos_exclude_full --ref_part    # reference aussi pour les classes partielles
echo "[$(date +%H:%M)] fini"
