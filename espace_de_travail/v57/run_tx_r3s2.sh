#!/bin/bash
# v57 : "professeur" re-appris UNIQUEMENT sur les images fabriquees a partir de texte (3 derniers blocs remis a zero, graine 2) :
# il n'a jamais revu les images a oublier -> il y repond "par generalisation", comme un gold. ~6 min. N'ecrit que dans v57/ (+ cache v45).
cd /p/scratch/training2666/huang25/Epi84/unlearning/v57 || exit 1
until [ -e gen/txt_all.pt ]; do sleep 5; done
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python /p/scratch/training2666/huang25/Epi84/unlearning/v53/redistill.py --data /p/scratch/training2666/huang25/Epi84/unlearning/v57/gen/txt_all.npy --reinit 3 --seed 2 --out /p/scratch/training2666/huang25/Epi84/unlearning/v57/subs/tx_r3s2.pt > logs/tx_r3s2.log 2>&1 || echo "ECHEC tx_r3s2"
cd /p/scratch/training2666/huang25/Epi84/unlearning/v45 && /p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python eval_models.py tx_r3s2=/p/scratch/training2666/huang25/Epi84/unlearning/v57/subs/tx_r3s2.pt >> logs/bilan.log 2>&1
echo fini
