#!/bin/bash
# v45 : cache des sorties de e_50b, e_t2_50, e_t2_r, e_70, p_8_8, p_7_6. ~4 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py e_50b=v47/subs/e_50b.pt e_t2_50=v47/subs/e_t2_50.pt e_t2_r=v47/subs/e_t2_r.pt e_70=v47/subs/e_70.pt p_8_8=v47/subs/p_8_8.pt p_7_6=v47/subs/p_7_6.pt >> logs/bilan.log 2>&1; echo fini >> logs/bilan.log
