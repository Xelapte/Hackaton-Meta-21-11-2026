#!/bin/bash
# v60 : U-Net empileur avec en plus les cartes multi-echelles (ms_*.npz de sig_ms.py), 5 plis geles x 2 graines. ~6 min GPU. N'ecrit que dans v60/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v60 || exit 1
export PYTHONDONTWRITEBYTECODE=1
if [ -f ms_pixelseal.npz ] && [ -f ms_trustmark.npz ]; then
  /p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python unet2.py --tag c2 --corr 0 --ms 1 --epochs 40 --seeds 2 > logs/unet_c2.log 2>&1 || echo "ECHEC unet_c2"
else
  echo "ms_*.npz absents : unet_c2 non lance" > logs/unet_c2.log
fi
echo fini
