#!/bin/bash
# v60 : U-Net empileur (10 cartes + plans boite, perte mIoU des 2 classes), 5 plis geles x 2 graines. ~6 min GPU. N'ecrit que dans v60/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v60 || exit 1
export PYTHONDONTWRITEBYTECODE=1
/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task/.venv/bin/python unet2.py --tag c0 --corr 0 --epochs 40 --seeds 2 > logs/unet_c0.log 2>&1 || echo "ECHEC unet_c0"
echo fini
