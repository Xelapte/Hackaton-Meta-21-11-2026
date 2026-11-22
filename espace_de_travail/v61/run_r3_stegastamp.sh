#!/bin/bash
# v61 : 3e passe (stegastamp) = on CONTINUE l'entrainement synthetique a partir des poids de la passe r2, puis CV 5 plis.
# Ne fait rien (sortie immediate, 0 s de GPU) si le fichier GO_R3_stegastamp n'existe pas : la place dans la file est reservee
# sans cout tant que la passe n'est pas decidee. Ecrit uniquement dans v61/.
V=/p/scratch/training2666/huang25/Epi84/unlearning/v61
[ -f $V/GO_R3_stegastamp ] || { echo "passe r3 non demandee pour stegastamp : rien a faire"; exit 0; }
cd $V && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth_new.py --method stegastamp --tag r3 --init $V/poids_stegastamp_r2.pt --synth_secs 330 --ft_secs 30 --seed 1 > $V/logs/run_r3_stegastamp.log 2>&1
