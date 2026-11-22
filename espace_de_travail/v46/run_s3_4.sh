#!/bin/bash
# v46 : seules 0 et 38 retirees ; oubli des autres classes plus leger ou cible sur les classes voisines. Variante s3_4 (delta_part 0 --sib_k 3 --sib_delta 4 --sib_full 3). ~3 min. N'ecrit que dans v46/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v46 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 0 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes 0,38 --sib_k 3 --sib_delta 4 --sib_full 3 --out subs/s3_4.pt > logs/s3_4.log 2>&1 || echo "ECHEC s3_4"
echo fini
