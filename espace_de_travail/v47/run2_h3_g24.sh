#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init subs/h4_all.pt --sac subs/h7_all_sac.pt --sac_frac 0.03 --margin 23.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/h3_g24.pt). ~5 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/h4_all.pt --sac subs/h7_all_sac.pt --sac_frac 0.03 --margin 23.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/h3_g24.pt > logs/e2_h3_g24.log 2>&1 || echo "ECHEC h3_g24"
echo fini
