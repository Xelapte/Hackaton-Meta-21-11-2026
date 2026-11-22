#!/bin/bash
# v45 : pour une liste de modeles "tag=chemin", met en cache sorties exactes/copies + vues fortes. Usage (via la file GPU) : bash bilan.sh tag=chemin ...
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
PY=/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python
$PY eval_models.py "$@" >> logs/bilan.log 2>&1; $PY eval_vues.py "$@" >> logs/bilan.log 2>&1
echo fini
