#!/bin/bash
# v55 : essai du generateur de variantes sur 200 images, deux forces. ~3 min. N'ecrit que dans v55/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v55 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_var.py 1000 1200 0.5 1 /p/scratch/training2666/huang25/Epi84/unlearning/v55/gen/essai_050.npy /p/scratch/training2666/huang25/Epi84/unlearning/v55/gen/essai_050.jpg > logs/test_050.log 2>&1 || echo "ECHEC 050"
echo fini
