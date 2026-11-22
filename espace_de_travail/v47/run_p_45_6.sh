#!/bin/bash
# v47 : etape 1 (classes partielles delta 4.5, 9 classes lourdes delta 6, liste fragile des classes lourdes) puis etape 2 (252 images lourdes fragiles a loss 16.5, copies +2.5). ~9 min. N'ecrit que dans v47/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v47 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 4.5 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes "" --mid_classes 0,38,6,7,8,29,30,31,33 --delta_mid 6 --sac_frac 0.072 --sac_scope heavy --sac_margin 16 --out subs/s1_45_6.pt > logs/s1_45_6.log 2>&1 || { echo "ECHEC etape 1 45_6"; exit 1; }
$B/.venv/bin/python etape2.py --init subs/s1_45_6.pt --sac subs/s1_45_6_sac.pt --margin 16.5 --copy_extra 2.5 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/p_45_6.pt > logs/e2_p_45_6.log 2>&1 || echo "ECHEC etape 2 45_6"
echo fini
