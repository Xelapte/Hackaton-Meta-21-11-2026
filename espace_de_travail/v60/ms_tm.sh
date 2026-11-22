#!/bin/bash
# v60 : cartes « logits par position » multi-echelles, TrustMark (train+test). ~2-4 min GPU. N'ecrit que dans v60/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v60 || exit 1
export PYTHONDONTWRITEBYTECODE=1
/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python sig_ms.py trustmark > logs/ms_tm.log 2>&1 || echo "ECHEC ms_tm"
echo fini
