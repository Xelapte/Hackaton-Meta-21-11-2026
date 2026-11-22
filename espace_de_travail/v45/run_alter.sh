#!/bin/bash
# v45 : robustesse du sacrifice a d'autres types de modifications (flou, rotation, resolution...). ~2 min. N'ecrit que dans v45/logs.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python alter.py record=v47/subs/hb_comb.pt base0=v47/subs/base0.pt > logs/alter.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
