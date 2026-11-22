#!/bin/bash
# v22 : maquette realiste de l'epreuve (voir replica3.py). ~15 min sur 1 GPU. N'ecrit que dans v22/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v22 || exit 1
mkdir -p logs
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica3.py > logs/replica3.log 2>&1 || echo "ECHEC (voir logs/replica3.log)"
echo "fini"
