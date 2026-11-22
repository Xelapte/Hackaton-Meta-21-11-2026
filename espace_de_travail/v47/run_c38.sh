#!/bin/bash
# v47 : images "sacrifiees" choisies une par une (les plus fragiles) au lieu de classes entieres. Variante c38. ~3 min. N'ecrit que dans v47/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v47 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes 38 --out subs/c38.pt > logs/c38.log 2>&1 || echo "ECHEC c38"
echo fini
