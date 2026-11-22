#!/bin/bash
# v46 : seules 0 et 38 retirees ; oubli des autres classes plus leger ou cible sur les classes voisines. Variante d2 (delta_part 2 ). ~3 min. N'ecrit que dans v46/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v46 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 2 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes 0,38  --out subs/d2.pt > logs/d2.log 2>&1 || echo "ECHEC d2"
echo fini
