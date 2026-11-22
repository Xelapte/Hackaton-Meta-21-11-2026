#!/bin/bash
# v70 : etape 2 seule sur la meilleure base (v47/s1_7_7, record p_7_7 = 0.3144 avec marge 16.5 / copies +2.5) : marge 17.2, copies +3.2.
# Motif : sur deux autres bases, marge 17.2 / +3.2 a fait +0.011 (e_50 -> e_50b) et +0.012 (ep_7_7 -> ep_7_7b). ~5 min. N'ecrit que dans v70/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v70 || exit 1
$B/.venv/bin/python $B/v47/etape2.py --init $B/v47/subs/s1_7_7.pt --sac $B/v47/subs/lourd_comb_sac.pt --margin 17.2 --copy_extra 3.2 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out $B/v70/subs/p77_m172.pt > logs/e2_p77_m172.log 2>&1 || echo "ECHEC etape 2 p77_m172"
echo fini
