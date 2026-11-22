#!/bin/bash
# v51 : recette du record (0 et 38 seules retirees), mais le modele doit etre MOINS sur de lui sur les copies modifiees que sur l'image exacte. Variante c05_d2 (aug_train 0.5 --delta_copy 2). ~3 min. N'ecrit que dans v51/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v51 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.5 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes 0,38 --delta_copy 2 --out subs/c05_d2.pt > logs/c05_d2.log 2>&1 || echo "ECHEC c05_d2"
echo fini
