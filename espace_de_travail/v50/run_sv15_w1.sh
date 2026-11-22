#!/bin/bash
# v50 : recette du record (0 et 38 seules retirees) + ancrage sur des vues FORTES des images a oublier. Variante sv15_w1 (--view 1.5 --w_view 1). ~4 min. N'ecrit que dans v50/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v50 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes 0,38 --view 1.5 --w_view 1 --out subs/sv15_w1.pt > logs/sv15_w1.log 2>&1 || echo "ECHEC sv15_w1"
echo fini
