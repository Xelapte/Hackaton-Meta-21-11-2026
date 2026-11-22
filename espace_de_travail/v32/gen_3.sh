#!/bin/bash
# v32 : generation d'images synthetiques pour les classes 75 a 99 (40 par classe). ~5 min. N'ecrit que dans v32/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v32 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen.py 75 100 40 /p/scratch/training2666/huang25/Epi84/unlearning/v32/syn/syn_3.pt > logs/gen_3.log 2>&1 || echo "ECHEC gen 3"
echo fini
