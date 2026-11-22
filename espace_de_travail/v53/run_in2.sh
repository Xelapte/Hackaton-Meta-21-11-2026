#!/bin/bash
# v53 : professeur "de-memorise" supplementaire (--reinit 2 --self_frac 0.6 --seed 5) pour mieux reperer les images fragiles. ~6 min. N'ecrit que dans v53/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v53 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python redistill.py --reinit 2 --self_frac 0.6 --seed 5 --out /p/scratch/training2666/huang25/Epi84/unlearning/v53/subs/in2.pt > logs/in2.log 2>&1 || echo "ECHEC in2"
echo fini
