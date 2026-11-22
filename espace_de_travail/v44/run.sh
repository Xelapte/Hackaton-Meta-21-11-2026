#!/bin/bash
# v44 : forme de la cible (marge plafonnee, temperature, ancrage) dans la maquette realiste. ~20 min. N'ecrit que dans v44/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v44 || exit 1
NB=10 /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica9.py > logs/replica9_nb10.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
