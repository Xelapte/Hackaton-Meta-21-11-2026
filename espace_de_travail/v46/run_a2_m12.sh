#!/bin/bash
# v46 : recette du record (ancrage photos), seules les classes 0 et 38 traitees comme retirees. Variante a2_m12 (--mid_classes 6,7,8,29,30,31,33 --delta_mid 12). ~5 min. N'ecrit que dans v46/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v46 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes 0,38 --mid_classes 6,7,8,29,30,31,33 --delta_mid 12 --out subs/a2_m12.pt > logs/a2_m12.log 2>&1 || echo "ECHEC a2_m12"
echo fini
