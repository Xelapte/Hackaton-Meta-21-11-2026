#!/bin/bash
# v63 : TrustMark, suite de la chaine L5 -> L11 -> T17 : 1000 s de synthetique en plus depuis les poids L11 (fin vers 11h36). Ecrit uniquement dans v63/.
V=/p/scratch/training2666/huang25/Epi84/unlearning
INIT=$V/v63/poids_trustmark_L11.pt; [ -f $INIT ] || INIT=$V/v62/poids_trustmark_L5.pt
cd $V/v63 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method trustmark --tag T17 --init $INIT --synth_secs 1000 --ft_secs 45 --cap 3000 --eval_every 600 --no_dry --seed 17 > $V/v63/logs/run_T17.log 2>&1
