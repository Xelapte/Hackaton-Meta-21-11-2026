#!/bin/bash
# v45 : traits de l'original sur les images a oublier (exactes + vues). ~2 min. N'ecrit que dans v45/ (petit fichier).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python feats.py > logs/feats.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
