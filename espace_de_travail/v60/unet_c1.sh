#!/bin/bash
# v60 : U-Net empileur avec en plus la correlation re-tatouage (corr_*.npz de sig_corr.py), 5 plis geles x 2 graines. ~6 min GPU. N'ecrit que dans v60/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v60 || exit 1
export PYTHONDONTWRITEBYTECODE=1
if [ -f corr_pixelseal.npz ] && [ -f corr_stegastamp.npz ] && [ -f corr_trustmark.npz ]; then
  /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python unet2.py --tag c1 --corr 1 --epochs 40 --seeds 2 > logs/unet_c1.log 2>&1 || echo "ECHEC unet_c1"
else
  echo "corr_*.npz absents : unet_c1 non lance" > logs/unet_c1.log
fi
echo fini
