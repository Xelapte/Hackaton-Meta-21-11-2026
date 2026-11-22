#!/bin/bash
# v45 : cache des sorties des modeles v46 (d0, d2, d4, s3_4...). ~1 min. N'ecrit que dans v45/ (petits fichiers).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py $(for f in ../v46/subs/*.pt; do n=$(basename $f .pt); case $n in *_a[0-9]*) ;; *) echo "$n=v46/subs/$n.pt";; esac; done | tr '\n' ' ') > logs/eval3.log 2>&1 || echo "ECHEC (voir logs)"
echo fini
