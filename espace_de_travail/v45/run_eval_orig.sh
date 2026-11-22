#!/bin/bash
# v45 : cache des sorties du modele original (exactes + copies). ~1 min. N'ecrit que dans v45/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py orig=model.pt > logs/eval_orig.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
