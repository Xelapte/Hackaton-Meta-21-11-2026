#!/bin/bash
# v35 : re-distillation 40 000 photos, variante r2x (--reinit 2 --steps 12000). Attend la fin de la preparation. N'ecrit que dans v35/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v35 || exit 1
while [ ! -f /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_unl_40k_224.npy ]; do sleep 15; done
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python redistill.py --reinit 2 --steps 12000 --data /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_unl_40k_224.npy --out /p/scratch/training2666/huang25/Epi84/unlearning/v35/subs/rd40_r2x.pt > logs/rd40_r2x.log 2>&1 || echo "ECHEC r2x"
echo fini
