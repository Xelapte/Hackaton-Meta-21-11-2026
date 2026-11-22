#!/bin/bash
# v45 : sorties de plusieurs modeles sur des vues fortes (retain local). ~3 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_vues.py orig=model.pt ext_wr1=v36/subs/ext_wr1.pt a2=v46/subs/a2.pt d4=v46/subs/d4.pt s3_4=v46/subs/s3_4.pt rd_r2=v34/subs/rd_r2.pt aucune=v31/subs/aucune.pt ext_wr3=v36/subs/ext_wr3.pt d0=v46/subs/d0.pt e1=v16/subs/e1.pt e2=v16/subs/e2.pt e3=v16/subs/e3.pt > logs/vues.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
