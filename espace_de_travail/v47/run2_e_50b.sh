#!/bin/bash
# v47 etape 2 : enfonce la loss des images sacrifiees (--init subs/s1_e50.pt --sac subs/lourd_comb_sac.pt --margin 17.2 --copy_extra 3.2 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/e_50b.pt). ~5 min. N'ecrit que dans v47/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v47 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape2.py --init subs/s1_e50.pt --sac subs/lourd_comb_sac.pt --margin 17.2 --copy_extra 3.2 --aug_sac 0.7 --lr 1.5e-5 --w_sac 0.3 --w_keep 4 --steps 700 --out subs/e_50b.pt > logs/e2_e_50b.log 2>&1 || echo "ECHEC e_50b"
echo fini
