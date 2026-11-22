#!/bin/bash
# v63 : derniere passe courte pixelseal (graine 25) : 300 s de synthetique depuis v63/poids_pixelseal_L9.pt, affinage CV 40 s / pli (membre d'ensemble). Fin ~11h37.
V=/p/scratch/training2666/huang25/Epi84/unlearning
cd $V/v63 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method pixelseal --tag Q25 --init $V/v63/poids_pixelseal_L9.pt --synth_secs 300 --ft_secs 40 --cap 3000 --eval_every 600 --no_dry --seed 25 > $V/v63/logs/run_Q25.log 2>&1
