#!/bin/bash
# v39 : pipeline "professeur dememorise" dans la maquette, avec 10 blocs figes. ~20 min. N'ecrit que dans v39/logs (aucun modele).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v39 || exit 1
NB=10 /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica7.py > logs/replica7_nb10.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
