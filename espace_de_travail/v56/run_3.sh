#!/bin/bash
# v56 : indice "temps d'oubli" (SSFT), 1 tirage (graine 7). ~3 min. N'ecrit que dans v56/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v56 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python ssft.py 7 /p/scratch/training2666/huang25/Epi84/unlearning/v56/ssft_7.pt > logs/ssft_7.log 2>&1 || echo "ECHEC 7"
echo fini
