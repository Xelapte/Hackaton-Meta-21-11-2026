#!/bin/bash
# v32 : effacer puis reparer sur le vrai modele, variante n20 (--noise 2.0). ~5 min. N'ecrit que dans v32/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v32 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python er.py --noise 2.0 --out /p/scratch/training2666/huang25/Epi84/unlearning/v32/subs/er_n20.pt > logs/er_n20.log 2>&1 || echo "ECHEC er n20"
echo fini
