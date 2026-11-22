#!/bin/bash
# v61 : segmentation apprise sur donnees synthetiques, methode pixelseal (pre-entrainement synthetique + CV 5 plis sur le vrai train).
# Attend au plus 3 min que le code soit pret (fichier READY), puis lance seg_synth.py. Ecrit uniquement dans v61/.
V=/p/scratch/training2666/huang25/Epi84/unlearning/v61
n=0; until [ -f $V/READY ] || [ $n -ge 90 ]; do sleep 2; n=$((n+1)); done
[ -f $V/READY ] || { echo "code pas pret, abandon" > $V/logs/run_pixelseal.log; exit 3; }
cd $V && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth_new.py --method pixelseal --tag r2 > $V/logs/run_pixelseal.log 2>&1
