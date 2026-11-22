#!/bin/bash
# v45 : cache (sorties + vues fortes) de tous les modeles v52 deja enregistres. ~1 min par modele. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning || exit 1
A=""; for f in v52/subs/*.pt; do n=$(basename $f .pt); case $n in *_a[0-9]*|*_t0*|*_b[0-9]*) ;; *) A="$A $n=$f";; esac; done
bash v45/bilan.sh $A
