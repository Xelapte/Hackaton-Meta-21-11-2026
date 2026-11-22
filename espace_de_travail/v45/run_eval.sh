#!/bin/bash
# v45 : met en cache les sorties de plusieurs modeles deja entraines (aucun entrainement). ~3 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py ext_wr1=v36/subs/ext_wr1.pt ext_wr3=v36/subs/ext_wr3.pt t3_5=v40/subs/t3_5.pt t3_all=v40/subs/t3_all.pt t3_9=v40/subs/t3_9.pt rd_r2=v34/subs/rd_r2.pt rd40_r2=v35/subs/rd40_r2.pt rdai_r2=v38/subs/rdai_r2.pt > logs/eval.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
