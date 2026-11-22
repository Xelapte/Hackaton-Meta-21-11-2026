#!/bin/bash
# v32 : generation d'images synthetiques pour les classes 50 a 74 (40 par classe). ~5 min. N'ecrit que dans v32/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v32 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen.py 50 75 40 /p/scratch/training2666/huang25/Epi84/unlearning/v32/syn/syn_2.pt > logs/gen_2.log 2>&1 || echo "ECHEC gen 2"
echo fini
