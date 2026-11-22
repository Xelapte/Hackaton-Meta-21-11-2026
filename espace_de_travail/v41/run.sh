#!/bin/bash
# v41 : AMUN (exemples adverses) dans la maquette realiste a 10 blocs figes. ~25 min. N'ecrit que dans v41/logs (aucun modele).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v41 || exit 1
NB=10 /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica8.py > logs/replica8_nb10.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
