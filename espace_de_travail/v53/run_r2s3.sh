#!/bin/bash
# v53 : professeur "de-memorise" supplementaire (--reinit 2 --seed 3 --lr 3e-4) pour mieux reperer les images fragiles. ~5 min. N'ecrit que dans v53/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v53 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python redistill.py --reinit 2 --seed 3 --lr 3e-4 --out /p/scratch/training2666/huang25/Epi84/unlearning/v53/subs/r2s3.pt > logs/r2s3.log 2>&1 || echo "ECHEC r2s3"
echo fini
