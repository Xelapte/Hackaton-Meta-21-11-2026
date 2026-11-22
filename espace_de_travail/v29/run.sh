#!/bin/bash
# v29 : maquette realiste de l'epreuve (voir replica6.py). ~15 min sur 1 GPU. N'ecrit que dans v29/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v29 || exit 1
mkdir -p logs
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica6.py > logs/replica6.log 2>&1 || echo "ECHEC (voir logs/replica6.log)"
echo "fini"
