#!/bin/bash
# v54 etape 3 : l'eleve garde l'oubli du record sur l'image exacte et ses copies, et redevient l'original sur les vues tres modifiees (--view_lo 1.25 --view_hi 1.75 --w_view 1 --w_cut 0.5 --w_f 4 --margin 16.5 --w_depth 0). ~4 min. N'ecrit que dans v54/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v54 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape3.py --init ../v47/subs/hb_comb.pt --sac ../v47/subs/lourd_comb_sac.pt --view_lo 1.25 --view_hi 1.75 --w_view 1 --w_cut 0.5 --w_f 4 --margin 16.5 --w_depth 0 --out subs/q125.pt > logs/q125.log 2>&1 || echo "ECHEC q125"
echo fini
