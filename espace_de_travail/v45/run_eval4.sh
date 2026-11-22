#!/bin/bash
# v45 : cache des sorties des modeles v47 c0 et c38. ~1 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py c0=v47/subs/c0.pt c38=v47/subs/c38.pt > logs/eval4.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
