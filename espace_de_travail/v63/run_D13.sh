#!/bin/bash
# v63 : passe profonde stegastamp (graine 13) : 2700 s de synthetique en repartant de v62/poids_stegastamp_L7.pt, affinage CV 45 s / pli, TTA miroir a part.
# Ecrit uniquement dans v63/.
V=/p/scratch/training2666/huang25/Epi84/unlearning
cd $V/v63 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method stegastamp --tag D13 --init $V/v62/poids_stegastamp_L7.pt --synth_secs 2700 --ft_secs 45 --cap 3000 --eval_every 600 --no_dry --seed 13 > $V/v63/logs/run_D13.log 2>&1
