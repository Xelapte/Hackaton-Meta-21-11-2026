#!/bin/bash
# v63 : derniere passe courte stegastamp (graine 23) : 300 s de synthetique depuis v63/poids_stegastamp_D13.pt, affinage CV 40 s / pli (membre d'ensemble). Fin ~11h37.
V=/p/scratch/training2666/huang25/Epi84/unlearning
cd $V/v63 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method stegastamp --tag E23 --init $V/v63/poids_stegastamp_D13.pt --synth_secs 300 --ft_secs 40 --cap 3000 --eval_every 600 --no_dry --seed 23 > $V/v63/logs/run_E23.log 2>&1
