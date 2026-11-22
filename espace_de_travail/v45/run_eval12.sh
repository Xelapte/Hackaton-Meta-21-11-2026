#!/bin/bash
# v45 : cache des sorties de e_50, e_30, p_7_7, pc1. ~3 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py e_50=v47/subs/e_50.pt e_30=v47/subs/e_30.pt p_7_7=v47/subs/p_7_7.pt pc1=v47/subs/pc1.pt >> logs/bilan.log 2>&1; echo fini >> logs/bilan.log
