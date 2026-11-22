#!/bin/bash
# v35 : re-distillation avec 40 000 photos exterieures (COCO unlabeled, lues dans le zip, 1 seul fichier en sortie).
# Preparation (~6 min de CPU) puis 2 blocs remis a zero, 6000 pas (~11 min). N'ecrit que dans v35/ et ext/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v35 || exit 1
PY=/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python
[ -f /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_unl_40k_224.npy ] || $PY prep.py /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_unlabeled2017.zip /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_unl_40k_224.npy 40000 > logs/prep.log 2>&1 || echo "ECHEC prep"
$PY redistill.py --reinit 2 --steps 6000 --data /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_unl_40k_224.npy --out /p/scratch/training2666/huang25/Epi84/unlearning/v35/subs/rd40_r2.pt > logs/rd40_r2.log 2>&1 || echo "ECHEC rd40_r2"
echo fini
