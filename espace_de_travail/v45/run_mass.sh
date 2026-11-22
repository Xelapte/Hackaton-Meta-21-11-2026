#!/bin/bash
# v45 : ou va la probabilite du modele du record, classe par classe. ~2 min. N'ecrit que dans v45/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python diag_mass.py > logs/mass.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
