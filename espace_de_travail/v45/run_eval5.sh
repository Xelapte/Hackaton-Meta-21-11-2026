#!/bin/bash
# v45 : cache des sorties des modeles v47 (sacrifice image par image) + vues fortes. ~3 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning || exit 1
bash v45/bilan.sh h7_all_g16=v47/subs/h7_all_g16.pt h7_heavy_g16=v47/subs/h7_heavy_g16.pt h4_all_g22=v47/subs/h4_all_g22.pt h7_all=v47/subs/h7_all.pt
