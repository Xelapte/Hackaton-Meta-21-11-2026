#!/bin/bash
# v52 : recette du record v36/ext_wr1 (9 classes en delta 20, ancrage photos) avec variantes : sl20 (--k_aug 16 --delta_slope 20). ~3 min. N'ecrit que dans v52/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v52 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --k_aug 16 --delta_slope 20 --out subs/sl20.pt > logs/sl20.log 2>&1 || echo "ECHEC sl20"
echo fini
