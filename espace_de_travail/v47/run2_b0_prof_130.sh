#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init subs/base0.pt --sac subs/liste_prof_sac.pt --sac_frac 0.0371 --margin 21.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/b0_prof_130.pt). ~5 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/base0.pt --sac subs/liste_prof_sac.pt --sac_frac 0.0371 --margin 21.5 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/b0_prof_130.pt > logs/e2_b0_prof_130.log 2>&1 || echo "ECHEC b0_prof_130"
echo fini
