#!/bin/bash
# v53 : professeur "de-memorise" supplementaire (--reinit 2 --self_frac 0.6 --self_lo 1.25 --self_hi 2.0 --seed 7) pour mieux reperer les images fragiles. ~6 min. N'ecrit que dans v53/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v53 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python redistill.py --reinit 2 --self_frac 0.6 --self_lo 1.25 --self_hi 2.0 --seed 7 --out /p/scratch/training2666/huang25/Epi84/unlearning/v53/subs/in2h.pt > logs/in2h.log 2>&1 || echo "ECHEC in2h"
echo fini
