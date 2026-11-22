#!/bin/bash
# v45 : cache des sorties des professeurs v53 et des nouveaux modeles v47. ~5 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning || exit 1
A="r2s1=v53/subs/r2s1.pt r3=v53/subs/r3.pt r2s3=v53/subs/r2s3.pt"; for n in hb_robprof hbg_robprof hh9_c2 hhg hb_trois; do [ -f v47/subs/$n.pt ] && A="$A $n=v47/subs/$n.pt"; done
cd v45 && /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py $A >> logs/bilan.log 2>&1; echo fini >> logs/bilan.log
