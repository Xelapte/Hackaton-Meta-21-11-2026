#!/bin/bash
# v42 : exemples adverses pour les images 1750 a 2625 du forget set (vrai modele). ~10-15 min. N'ecrit que dans v42/ (1 fichier de 260 Mo).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v42 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_adv.py 1750 2625 /p/scratch/training2666/huang25/Epi84/unlearning/v42/adv/adv_2.pt > logs/gen_2.log 2>&1 || echo "ECHEC gen 2"
echo fini
