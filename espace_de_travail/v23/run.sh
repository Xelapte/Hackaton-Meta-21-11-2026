#!/bin/bash
# v23 : maquette realiste de l'epreuve (voir replica4.py). ~15 min sur 1 GPU. N'ecrit que dans v23/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v23 || exit 1
mkdir -p logs
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python replica4.py > logs/replica4.log 2>&1 || echo "ECHEC (voir logs/replica4.log)"
echo "fini"
