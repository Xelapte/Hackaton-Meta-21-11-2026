#!/bin/bash
# v54 etape 3 : l'eleve garde l'oubli du record sur l'image exacte et ses copies, et redevient l'original sur les vues tres modifiees (--view_lo 1.0 --view_hi 1.5 --w_view 2 --copy_max 0.5). ~6 min. N'ecrit que dans v54/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v54 || exit 1
/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python etape3.py --init ../v47/subs/hb_comb.pt --sac ../v47/subs/lourd_comb_sac.pt --view_lo 1.0 --view_hi 1.5 --w_view 2 --copy_max 0.5 --out subs/r100.pt > logs/r100.log 2>&1 || echo "ECHEC r100"
echo fini
