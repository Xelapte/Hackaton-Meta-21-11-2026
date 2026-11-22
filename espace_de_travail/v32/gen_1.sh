#!/bin/bash
# v32 : generation d'images synthetiques pour les classes 25 a 49 (40 par classe). ~5 min. N'ecrit que dans v32/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v32 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python gen.py 25 50 40 /p/scratch/training2666/huang25/Epi84/unlearning/v32/syn/syn_1.pt > logs/gen_1.log 2>&1 || echo "ECHEC gen 1"
echo fini
