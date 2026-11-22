#!/bin/bash
# (modifie a 9h05 avant tout lancement : le noeud GPU s'est arrete a 8h57, rien n'avait tourne)
# 1) PRIORITE : generation texte -> image de v57 (partie 0), ~7 min. 2) puis un tirage SSFT (v56), ~3 min.
# N'ecrit que dans v57/ et v56/.
bash /p/scratch/training2666/huang25/Epi84/unlearning/v57/run_t0.sh
cd /p/scratch/training2666/huang25/Epi84/unlearning/v56 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python ssft.py 1 /p/scratch/training2666/huang25/Epi84/unlearning/v56/ssft_1.pt > logs/ssft_1.log 2>&1 || echo "ECHEC 1"
echo fini
