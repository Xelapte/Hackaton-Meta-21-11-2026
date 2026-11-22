#!/bin/bash
# v45 : cache des sorties des modeles v46 (et v31, v18). ~2 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py a2=v46/subs/a2.pt a2_m9=v46/subs/a2_m9.pt a2_m12=v46/subs/a2_m12.pt BCpart=v31/subs/BCpart.pt aucune=v31/subs/aucune.pt k16=v18/subs/k16_wr1_at025.pt > logs/eval2.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
