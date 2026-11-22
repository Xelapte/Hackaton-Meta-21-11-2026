#!/bin/bash
# v63 : test A/B de la duree d'affinage (StegaStamp) : memes poids synthetiques que v62 L7, AUCUN entrainement synthetique en plus,
# affinage CV de 135 s par pli au lieu de 45 s. Compare directement a v62 L7 (CV 0,7797). ~13 min GPU. Ecrit uniquement dans v63/.
V=/p/scratch/training2666/huang25/Epi84/unlearning
cd $V/v63 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method stegastamp --tag F135 --init $V/v62/poids_stegastamp_L7.pt --synth_secs 0 --ft_secs 135 --cap 3000 --no_dry --seed 7 > $V/v63/logs/run_ss_F135.log 2>&1
