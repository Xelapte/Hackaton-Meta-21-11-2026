#!/bin/bash
# v45 : cache des sorties de hb_comb, q125 et des listes d'ensemble. ~3 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py hb_comb=v47/subs/hb_comb.pt q125=v54/subs/q125.pt hb_ens3=v47/subs/hb_ens3.pt hb_ensVT=v47/subs/hb_ensVT.pt hb_comb_m19=v47/subs/hb_comb_m19.pt hb_tout6=v47/subs/hb_tout6.pt >> logs/bilan.log 2>&1; echo fini >> logs/bilan.log
