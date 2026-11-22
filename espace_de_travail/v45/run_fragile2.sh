#!/bin/bash
# v45 : autres indices de fragilite (bruit sur les poids, elagage, lecture anticipee). ~3 min. N'ecrit que dans v45/ (petit fichier).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python fragile2.py > logs/fragile2.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
