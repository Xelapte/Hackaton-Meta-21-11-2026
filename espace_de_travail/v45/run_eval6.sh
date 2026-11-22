#!/bin/bash
# v45 : cache des sorties des nouveaux modeles v47 (etape 2). ~3 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning || exit 1
bash v45/bilan.sh h3_g24=v47/subs/h3_g24.pt h5_g19=v47/subs/h5_g19.pt h4_k8=v47/subs/h4_k8.pt
