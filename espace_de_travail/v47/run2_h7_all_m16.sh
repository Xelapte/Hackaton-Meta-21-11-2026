#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init subs/h7_all.pt --sac subs/h7_all_sac.pt --margin 16 --out subs/h7_all_m16.pt). ~4 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/h7_all.pt --sac subs/h7_all_sac.pt --margin 16 --out subs/h7_all_m16.pt > logs/e2_h7_all_m16.log 2>&1 || echo "ECHEC h7_all_m16"
echo fini
