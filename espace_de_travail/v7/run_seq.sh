#!/bin/bash
# v7 "reparation du retain" : on part d'un modele v6 (classes entieres eteintes), on garde ses predictions
# sur le forget set, et on le recolle a l'original sur des mosaiques SANS les classes entieres.
# ~2 min par run + ~1 min d'eval. A deposer dans la file du GPU 1.
set -u
BASE=/p/scratch/training2666/huang25/Epi84/unlearning
cd $BASE/v7 || exit 1
mkdir -p subs logs
PY=$BASE/.venv/bin/python
run () {  # nom, puis arguments
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py "$@" --out subs/$N.pt > logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
C="--target init --tau 1.0 --k_aug 0 --mos_exclude_full"
run r_b8        $C --init $BASE/v6/subs/f20_p8_b8.pt  --epochs 3 --lr 2e-5
run r_b8_wr6    $C --init $BASE/v6/subs/f20_p8_b8.pt  --epochs 3 --lr 2e-5 --w_ret 6
run r_b8_e5     $C --init $BASE/v6/subs/f20_p8_b8.pt  --epochs 5 --lr 3e-5
run r_b12       $C --init $BASE/v6/subs/f20_p8_b12.pt --epochs 3 --lr 2e-5
echo "[$(date +%H:%M)] eval"
$PY eval.py $BASE/v6/subs/f20_p8_b8.pt $BASE/v6/subs/f20_p8_b12.pt \
  subs/r_b8.pt subs/r_b8_wr6.pt subs/r_b8_e5.pt subs/r_b12.pt \
  --betas 0 --out $BASE/v7/logs/eval.jsonl > logs/eval.log 2>&1 || echo "  -> ECHEC eval"
echo "[$(date +%H:%M)] fini"
