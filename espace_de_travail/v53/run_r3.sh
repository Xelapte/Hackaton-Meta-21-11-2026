#!/bin/bash
# v53 : professeur "de-memorise" supplementaire (--reinit 3 --seed 2) pour mieux reperer les images fragiles. ~5 min. N'ecrit que dans v53/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v53 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python redistill.py --reinit 3 --seed 2 --out /p/scratch/training2666/huang25/Epi84/unlearning/v53/subs/r3.pt > logs/r3.log 2>&1 || echo "ECHEC r3"
echo fini
