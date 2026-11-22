#!/bin/bash
# v47 etape 2 (version douce) : enfonce la loss des images sacrifiees (--init subs/h7_all.pt --sac subs/h7_all_sac.pt --margin 15.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/h7_all_g16.pt). ~4 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/h7_all.pt --sac subs/h7_all_sac.pt --margin 15.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/h7_all_g16.pt > logs/e2_h7_all_g16.log 2>&1 || echo "ECHEC h7_all_g16"
echo fini
