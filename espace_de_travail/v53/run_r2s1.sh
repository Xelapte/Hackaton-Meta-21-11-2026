#!/bin/bash
# v53 : professeur "de-memorise" supplementaire (--reinit 2 --seed 1) pour mieux reperer les images fragiles. ~5 min. N'ecrit que dans v53/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v53 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python redistill.py --reinit 2 --seed 1 --out /p/scratch/training2666/huang25/Epi84/unlearning/v53/subs/r2s1.pt > logs/r2s1.log 2>&1 || echo "ECHEC r2s1"
echo fini
