#!/bin/bash
# v45 : diagnostic par classe sur le vrai modele (planches d'images, solidite par classe, classes voisines). ~3 min. N'ecrit que dans v45/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python diag_classes.py > logs/diag.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
