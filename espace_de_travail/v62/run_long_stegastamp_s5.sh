#!/bin/bash
# v62 : passe LONGUE (stegastamp, graine 5) = entrainement synthetique de 30 min (au lieu de 4-5 min en v61), reserve de 3000 images,
# en repartant des derniers poids v61 disponibles (r3 > r2 > r1), puis affinage CV 5 plis (memes plis que v61). ~38 min GPU.
# Ecrit uniquement dans v62/.
V=/p/scratch/training2666/huang25/Epi84/unlearning
INIT=""
for r in r3 r2 r1; do [ -f $V/v61/poids_stegastamp_$r.pt ] && { INIT=$V/v61/poids_stegastamp_$r.pt; break; }; done
echo "init = $INIT"
cd $V/v62 && /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python seg_synth.py --method stegastamp --tag L5 ${INIT:+--init $INIT} --synth_secs 1800 --ft_secs 45 --cap 3000 --eval_every 300 --seed 5 > $V/v62/logs/run_long_stegastamp_s5.log 2>&1
