#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init ../v36/subs/ext_wr1.pt --sac subs/h4_all_sac.pt --margin 21.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/h4_ext.pt). ~5 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init ../v36/subs/ext_wr1.pt --sac subs/h4_all_sac.pt --margin 21.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/h4_ext.pt > logs/e2_h4_ext.log 2>&1 || echo "ECHEC h4_ext"
echo fini
