#!/bin/bash
# v48 : maquette v2 (classes lourdes presque videes, il en reste 0.15). ~25 min. N'ecrit que dans v48/logs (aucun modele).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v48 || exit 1
KEEP=0.15 NB=10 /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica10.py > logs/replica10_keep0.15.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
