#!/bin/bash
# v61 : methode stegastamp, 2e passe (nouveau generateur synthetique + modele a deux echelles).
# Repart de l'encodeur deja pre-entraine en passe r1 s'il existe. Ecrit uniquement dans v61/.
V=/p/scratch/training2666/huang25/Epi84/unlearning/v61
cd $V && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth_new.py --method stegastamp --tag r2 --init $V/poids_stegastamp_r1.pt --synth_secs 330 --ft_secs 30 > $V/logs/run_stegastamp_r2.log 2>&1
