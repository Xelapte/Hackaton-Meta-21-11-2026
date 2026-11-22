#!/bin/bash
# v55 : essai du generateur de variantes sur 200 images, force 0.75. ~3 min. N'ecrit que dans v55/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v55 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_var.py 1000 1200 0.75 2 /p/scratch/training2666/huang25/Epi84/unlearning/v55/gen/essai_075.npy /p/scratch/training2666/huang25/Epi84/unlearning/v55/gen/essai_075.jpg > logs/test_075.log 2>&1 || echo "ECHEC 075"
echo fini
