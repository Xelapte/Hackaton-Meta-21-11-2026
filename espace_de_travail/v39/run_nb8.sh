#!/bin/bash
# v39 : pipeline "professeur dememorise" dans la maquette, avec 8 blocs figes. ~20 min. N'ecrit que dans v39/logs (aucun modele).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v39 || exit 1
NB=8 /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica7.py > logs/replica7_nb8.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
