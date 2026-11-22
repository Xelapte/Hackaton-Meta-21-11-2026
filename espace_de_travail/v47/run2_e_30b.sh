#!/bin/bash
# v47 etape 2 sur la base "cible d'ensemble" s1_e30 : 252 images lourdes (liste combinee) a loss 17.0, copies +3.0. ~5 min. N'ecrit que dans v47/ (+ cache v45).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/s1_e30.pt --sac subs/lourd_comb_sac.pt --margin 17.0 --copy_extra 3.0 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/e_30b.pt > logs/e2_e_30b.log 2>&1 || echo "ECHEC e_30b"
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 && /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py e_30b=/p/scratch/training2666/huang25/Epi84/unlearning/v47/subs/e_30b.pt >> logs/bilan.log 2>&1
echo fini
