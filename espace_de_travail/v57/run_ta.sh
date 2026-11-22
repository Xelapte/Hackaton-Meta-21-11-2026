#!/bin/bash
# v57 : recette du record (e_50b : cible d'ensemble + etape 2) mais ANCRE = images fabriquees a partir de texte (ta). ~16 min. N'ecrit que dans v57/ (+ cache v45).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v57 || exit 1
until [ -e gen/txt_all.pt ]; do sleep 5; done
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt_all.npy --ext_aug 0.5 --w_ret 1 --ext_frac 1.0 --full_classes "" --teacher_cache rd_r2,rd40_r2,rdai_r2,r3,r2s3 --teacher_mix 0.5 --out subs/s1_ta.pt > logs/s1_ta.log 2>&1 || { echo "ECHEC etape 1 ta"; exit 1; }
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/s1_ta.pt --sac /p/scratch/training2666/huang25/Epi84/unlearning/v47/subs/lourd_comb_sac.pt --anchor /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt_all.npy --anchor_aug 0.5 --margin 17.2 --copy_extra 3.2 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/ta.pt > logs/e2_ta.log 2>&1 || echo "ECHEC etape 2 ta"
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 && /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py ta=/p/scratch/training2666/huang25/Epi84/unlearning/v57/subs/ta.pt >> logs/bilan.log 2>&1
echo fini
