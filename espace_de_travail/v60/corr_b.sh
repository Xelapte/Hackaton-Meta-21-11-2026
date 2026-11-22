#!/bin/bash
# v60 : signal « re-tatouer et correler » (TrustMark, train+test). ~3-4 min GPU. N'ecrit que dans v60/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v60 || exit 1
export PYTHONDONTWRITEBYTECODE=1
/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python sig_corr.py trustmark > logs/corr_b.log 2>&1 || echo "ECHEC corr_b"
echo fini
