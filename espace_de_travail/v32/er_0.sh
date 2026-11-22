#!/bin/bash
# v32 : effacer puis reparer sur le vrai modele, variante n05 (--noise 0.5). ~5 min. N'ecrit que dans v32/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v32 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python er.py --noise 0.5 --out /p/scratch/training2666/huang25/Epi84/unlearning/v32/subs/er_n05.pt > logs/er_n05.log 2>&1 || echo "ECHEC er n05"
echo fini
