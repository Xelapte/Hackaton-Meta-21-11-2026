#!/bin/bash
# v57 : images fabriquees a partir de texte (objets 330 a 690 du vocabulaire, 20 par objet) + avis de l'original. ~7 min. N'ecrit que dans v57/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v57 || exit 1
[ -e /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt_1.npy ] && { echo "deja fait"; exit 0; }
[ -e /p/scratch/training2666/huang25/Epi84/unlearning/v57/logs/txt_1.log ] && { echo "deja lance ailleurs"; exit 0; }
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen_txt.py /p/scratch/training2666/huang25/Epi84/unlearning/v57/vocab.txt 330 690 20 101 /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt_1 > logs/txt_1.log 2>&1 || echo "ECHEC txt_1"
echo fini
