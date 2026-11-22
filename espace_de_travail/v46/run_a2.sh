#!/bin/bash
# v46 : recette du record (ancrage photos), seules les classes 0 et 38 traitees comme retirees. Variante a2 (). ~5 min. N'ecrit que dans v46/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v46 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes 0,38  --out subs/a2.pt > logs/a2.log 2>&1 || echo "ECHEC a2"
echo fini
