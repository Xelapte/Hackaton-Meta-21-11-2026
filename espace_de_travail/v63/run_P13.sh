#!/bin/bash
# v63 : PixelSeal, suite de la chaine L5 -> L9 -> P13 : 1500 s de synthetique en plus depuis les poids L9. Ecrit uniquement dans v63/.
V=/p/scratch/training2666/huang25/Epi84/unlearning
INIT=$V/v63/poids_pixelseal_L9.pt; [ -f $INIT ] || INIT=$V/v62/poids_pixelseal_L5.pt
cd $V/v63 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method pixelseal --tag P13 --init $INIT --synth_secs 1500 --ft_secs 45 --cap 3000 --eval_every 600 --no_dry --seed 13 > $V/v63/logs/run_P13.log 2>&1
