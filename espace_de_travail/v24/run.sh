#!/bin/bash
# v24 : maquette realiste de l'epreuve (voir replica5.py). ~15 min sur 1 GPU. N'ecrit que dans v24/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v24 || exit 1
mkdir -p logs
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica5.py > logs/replica5.log 2>&1 || echo "ECHEC (voir logs/replica5.log)"
echo "fini"
