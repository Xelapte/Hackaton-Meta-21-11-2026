#!/bin/bash
# v45 : fragilite robuste (original, 48 vues x 4 forces). ~8 min. N'ecrit que dans v45/ (petit fichier).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python marges.py > logs/marges.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
