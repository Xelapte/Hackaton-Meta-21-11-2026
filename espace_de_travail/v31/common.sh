# v31 : meme recette que v30 (record), avec d'autres choix de classes "entieres". Appele par run_a.sh / run_b.sh.
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v31 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {  # nom, liste des classes entieres ("" = aucune)
  N=$1; FC=$2
  echo "[$(date +%H:%M)] $N : classes entieres = {$FC}"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --w_ret 1 --aug_train 0.25 --full_classes "$FC" --out subs/$N.pt > logs/$N.log 2>&1 || { echo "  -> ECHEC $N"; return; }
  if [ -n "$FC" ]; then $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt $FC >> logs/$N.log 2>&1; fi
}
