#!/bin/bash
# Sweep v5 sequentiel sur 1 GPU (~2-3 min par run). A deposer dans la file du GPU 1.
# delta different pour les 9 classes supprimees en entier (delta_full) et les classes partielles (delta_part).
# Grille ajustee apres v4 : l'eleve n'atteint pas les cibles fortes (ancrage mosaiques) -> on teste
# l'option mosaiques (mos_full_delta) et un ancrage plus faible (w_ret 1).
set -u
BASE=/p/scratch/training2666/huang25/Epi84/unlearning
cd $BASE/v5 || exit 1
mkdir -p subs logs
PY=$BASE/.venv/bin/python
run () {  # nom, puis arguments
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py "$@" --out subs/$N.pt > logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
C="--target delta --tau 1.0 --k_aug 0"
run f20_p8         $C --delta_full 20 --delta_part 8
run f20_p10        $C --delta_full 20 --delta_part 10
run f20_p8_mos4    $C --delta_full 20 --delta_part 8 --mos_full_delta 4
run f20_p8_wr1     $C --delta_full 20 --delta_part 8 --w_ret 1
echo "[$(date +%H:%M)] fini"
