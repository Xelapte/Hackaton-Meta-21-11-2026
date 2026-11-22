#!/bin/bash
# v21 : maquette realiste de l'epreuve (voir replica2.py). ~15 min sur 1 GPU. N'ecrit que dans v21/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v21 || exit 1
mkdir -p logs
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica2.py > logs/replica2.log 2>&1 || echo "ECHEC (voir logs/replica2.log)"
echo "fini"
