#!/bin/bash
# v34 : preparation des images exterieures (COCO val, lues dans le zip, 1 seul fichier en sortie), puis re-distillation
# du vrai modele : 2 puis 4 derniers blocs remis a zero, 2000 pas (~12 min chacun). N'ecrit que dans v34/ et ext/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v34 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
[ -f $B/ext/coco_val_224.npy ] || $PY prep.py $B/ext/coco_val2017.zip $B/ext/coco_val_224.npy > logs/prep.log 2>&1 || echo "ECHEC prep"
echo "[$(date +%H:%M)] r2"; $PY redistill.py --reinit 2 --out $B/v34/subs/rd_r2.pt > logs/rd_r2.log 2>&1 || echo "ECHEC r2"
echo "[$(date +%H:%M)] r4"; $PY redistill.py --reinit 4 --out $B/v34/subs/rd_r4.pt > logs/rd_r4.log 2>&1 || echo "ECHEC r4"
echo "[$(date +%H:%M)] fini"
