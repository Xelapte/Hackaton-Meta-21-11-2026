#!/bin/bash
# v47 : etape 1 avec cible "d'ensemble" (--teacher_mix 0.5 --teacher_T 2) puis etape 2 (liste combinee, 252 images lourdes a loss 17, copies +3). ~9 min. N'ecrit que dans v47/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v47 || exit 1
$B/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes "" --teacher_cache rd_r2,rd40_r2,rdai_r2,r3,r2s3 --teacher_mix 0.5 --teacher_T 2 --out subs/s1_et2_50.pt > logs/s1_et2_50.log 2>&1 || { echo "ECHEC etape 1 et2_50"; exit 1; }
$B/.venv/bin/python etape2.py --init subs/s1_et2_50.pt --sac subs/lourd_comb_sac.pt --margin 17 --copy_extra 3 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/e_t2_50.pt > logs/e2_e_t2_50.log 2>&1 || echo "ECHEC etape 2 et2_50"
echo fini
