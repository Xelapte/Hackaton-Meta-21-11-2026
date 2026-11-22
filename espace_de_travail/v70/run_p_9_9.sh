#!/bin/bash
# v70 : recette du record p_7_7 (v47, 0.3144) avec delta_part 9, delta_mid 9, cible d'ensemble=0, etape 2 marge 16 copies +2.5. ~9 min. N'ecrit que dans v70/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v70 || exit 1
[ -e $B/v70/subs/p_9_9.pt ] && { echo "deja fait"; exit 0; }
$B/.venv/bin/python $B/v47/unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 9 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor $B/ext/coco_val_224.npy --w_ret 1 --ext_frac 1.0 --full_classes "" --mid_classes 0,38,6,7,8,29,30,31,33 --delta_mid 9  --out $B/v70/subs/s1_p_9_9.pt > logs/s1_p_9_9.log 2>&1 || { echo "ECHEC etape 1 p_9_9"; exit 1; }
$B/.venv/bin/python $B/v47/etape2.py --init $B/v70/subs/s1_p_9_9.pt --sac $B/v47/subs/lourd_comb_sac.pt --margin 16 --copy_extra 2.5 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out $B/v70/subs/p_9_9.pt > logs/e2_p_9_9.log 2>&1 || echo "ECHEC etape 2 p_9_9"
echo fini
