#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init subs/h4_all.pt --sac subs/h4_all_sac.pt --margin 22 --out subs/h4_all_m22.pt). ~4 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/h4_all.pt --sac subs/h4_all_sac.pt --margin 22 --out subs/h4_all_m22.pt > logs/e2_h4_all_m22.log 2>&1 || echo "ECHEC h4_all_m22"
echo fini
