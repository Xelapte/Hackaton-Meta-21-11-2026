#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init subs/base0.pt --sac subs/lourd_rob_sac.pt --margin 16.5 --copy_extra 2.5 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/hb_rob.pt). ~5 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/base0.pt --sac subs/lourd_rob_sac.pt --margin 16.5 --copy_extra 2.5 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/hb_rob.pt > logs/e2_hb_rob.log 2>&1 || echo "ECHEC hb_rob"
echo fini
