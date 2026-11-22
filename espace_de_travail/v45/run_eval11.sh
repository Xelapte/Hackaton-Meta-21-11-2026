#!/bin/bash
# v45 : cache des sorties des nouveaux modeles v47 (p_6_9, p_6_12, p_7_7, e_50, e_30). ~4 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
A=""; for n in p_6_9 p_6_12 p_7_7 e_50 e_30 hh9_c2 hhg; do [ -f ../v47/subs/$n.pt ] && A="$A $n=v47/subs/$n.pt"; done
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py $A >> logs/bilan.log 2>&1; echo fini >> logs/bilan.log
