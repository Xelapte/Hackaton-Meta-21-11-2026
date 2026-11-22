#!/bin/bash
# v57 : 2e tour d'images fabriquees, cible par classe (classes 0 a 50, 150 par classe), filtre "bonne classe + confiance >= 0.6". ~5 min. N'ecrit que dans v57/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v57 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_txt2.py 0 50 150 200 /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt2_0 > logs/txt2_0.log 2>&1 || echo "ECHEC txt2_0"
echo fini
