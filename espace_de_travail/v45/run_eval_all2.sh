#!/bin/bash
# v45 : cache des sorties de toutes les anciennes soumissions encore sur disque (lot 2). ~4 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py $(cat a_evaluer_part02 | tr '\n' ' ') > logs/eval_all2.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
