#!/bin/bash
# v45 : cache des sorties des nouveaux modeles v47. ~5 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning || exit 1
A=""; for n in p_5_65 p_5_9 p_45_6 hb_prof hb_comb hb_l1 hb_rob hh4_c3 hh9_c2 hhg; do [ -f v47/subs/$n.pt ] && A="$A $n=v47/subs/$n.pt"; done
bash v45/bilan.sh $A
