#!/bin/bash
# v49 : maquette v2, etude du RETAIN (ancrage sur vues fortes des images a oublier). ~25 min. N'ecrit que dans v49/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v49 || exit 1
KEEP=0.15 NB=10 /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica11.py > logs/replica11.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
