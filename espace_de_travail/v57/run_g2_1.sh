#!/bin/bash
# v57 : 2e tour d'images fabriquees, cible par classe (classes 50 a 100, 150 par classe), filtre "bonne classe + confiance >= 0.6". ~5 min. N'ecrit que dans v57/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v57 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_txt2.py 50 100 150 201 /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt2_1 > logs/txt2_1.log 2>&1 || echo "ECHEC txt2_1"
echo fini
