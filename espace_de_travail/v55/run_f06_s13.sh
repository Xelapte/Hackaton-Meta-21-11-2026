#!/bin/bash
# v55 : variantes des 3500 images a oublier (force 0.6, graine 13) + avis de l'original. ~6 min. N'ecrit que dans v55/ (1 fichier de 527 Mo).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v55 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_var.py 0 3500 0.6 13 /p/scratch/training2666/huang25/Epi84/unlearning/v55/gen/var_f06_s13.npy /p/scratch/training2666/huang25/Epi84/unlearning/v55/gen/var_f06_s13.jpg > logs/f06_s13.log 2>&1 || echo "ECHEC f06_s13"
echo fini
