#!/bin/bash
# v45 : cache des sorties de toutes les anciennes soumissions encore sur disque (lot 0). ~4 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py $(cat a_evaluer_part00 | tr '\n' ' ') > logs/eval_all0.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
