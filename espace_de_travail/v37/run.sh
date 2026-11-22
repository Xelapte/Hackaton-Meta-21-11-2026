#!/bin/bash
# v37 : diagnostic sur les copies modifiees (aucun entrainement, ~2 min). N'ecrit que dans v37/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v37 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python diag.py > logs/diag.log 2>&1 || echo "ECHEC"
echo fini
