#!/bin/bash
# v32 : effacer puis reparer sur le vrai modele, variante r2 (--reinit 2 --steps 1000 --lr 2e-4). ~5 min. N'ecrit que dans v32/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v32 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python er.py --reinit 2 --steps 1000 --lr 2e-4 --out /p/scratch/training2666/huang25/Epi84/unlearning/v32/subs/er_r2.pt > logs/er_r2.log 2>&1 || echo "ECHEC er r2"
echo fini
