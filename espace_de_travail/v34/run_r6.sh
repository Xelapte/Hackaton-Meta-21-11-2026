#!/bin/bash
# v34 : re-distillation sur images exterieures, variante r6 (--reinit 6). N'ecrit que dans v34/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v34 || exit 1
while [ ! -f /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_val_224.npy ]; do sleep 10; done
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python redistill.py --reinit 6 --out /p/scratch/training2666/huang25/Epi84/unlearning/v34/subs/rd_r6.pt > logs/rd_r6.log 2>&1 || echo "ECHEC r6"
echo fini
