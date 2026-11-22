#!/bin/bash
# v20 : maquette locale de l'epreuve (voir replica.py). ~15-20 min sur 1 GPU. N'ecrit que dans v20/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v20 || exit 1
mkdir -p logs
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica.py > logs/replica.log 2>&1 || echo "ECHEC (voir logs/replica.log)"
echo "fini"
