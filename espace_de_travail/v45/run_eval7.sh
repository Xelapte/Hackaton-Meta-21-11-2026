#!/bin/bash
# v45 : cache des sorties des nouveaux modeles v47 (listes alternatives, sacrifice gradue, classes lourdes). ~5 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning || exit 1
A=""; for n in b0_prof_130 b0_comb_130 b0_vues_130 gr3 gr2 hh7_c3 hh5_c3 hh4_c3 h5_c3 h4_c3 base0; do [ -f v47/subs/$n.pt ] && A="$A $n=v47/subs/$n.pt"; done
bash v45/bilan.sh $A
