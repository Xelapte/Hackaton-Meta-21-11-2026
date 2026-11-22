#!/bin/bash
# v63 : passe longue pixelseal (graine 9) avec TTA miroir enregistree a part (train_oof_flip / test_cv_flip), 30 min de synthetique,
# reprise des meilleurs poids disponibles (v62 L5 > v61 r3 > r2). ~40 min GPU. Ecrit uniquement dans v63/.
V=/p/scratch/training2666/huang25/Epi84/unlearning
INIT=""
for f in $V/v62/poids_pixelseal_L5.pt $V/v61/poids_pixelseal_r3.pt $V/v61/poids_pixelseal_r2.pt; do [ -f $f ] && { INIT=$f; break; }; done
echo "init = $INIT"
cd $V/v63 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method pixelseal --tag L9 ${INIT:+--init $INIT} --synth_secs 1800 --ft_secs 45 --cap 3000 --eval_every 300 --seed 9 > $V/v63/logs/run_long_pixelseal_s9.log 2>&1
