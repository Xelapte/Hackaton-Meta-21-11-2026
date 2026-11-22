#!/bin/bash
# v57 : images fabriquees a partir de texte (objets 0 a 330 du vocabulaire, 20 par objet) + avis de l'original. ~7 min. N'ecrit que dans v57/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v57 || exit 1
[ -e /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt_0.npy ] && { echo "deja fait"; exit 0; }
[ -e /p/scratch/training2666/huang25/Epi84/unlearning/v57/logs/txt_0.log ] && { echo "deja lance ailleurs"; exit 0; }
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_txt.py /p/scratch/training2666/huang25/Epi84/unlearning/v57/vocab.txt 0 330 20 100 /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt_0 > logs/txt_0.log 2>&1 || echo "ECHEC txt_0"
echo fini
