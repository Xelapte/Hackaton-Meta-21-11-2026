#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init subs/h4_all.pt --sac subs/h7_all_sac.pt --profile 0.02:23,0.05:15 --copy_extra 2 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 800 --out subs/gr2.pt). ~5 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/h4_all.pt --sac subs/h7_all_sac.pt --profile 0.02:23,0.05:15 --copy_extra 2 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 800 --out subs/gr2.pt > logs/e2_gr2.log 2>&1 || echo "ECHEC gr2"
echo fini
