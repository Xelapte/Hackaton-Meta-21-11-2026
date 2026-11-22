#!/bin/bash
# v60 : signal « re-tatouer et correler » (PixelSeal + StegaStamp, train+test). ~3-4 min GPU. N'ecrit que dans v60/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v60 || exit 1
export PYTHONDONTWRITEBYTECODE=1
/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python sig_corr.py pixelseal,stegastamp > logs/corr_a.log 2>&1 || echo "ECHEC corr_a"
echo fini
