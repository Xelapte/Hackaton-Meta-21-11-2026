#!/bin/bash
# v40 : eleve = original + cible du professeur dememorise, variante t3_5. ~6 min. N'ecrit que dans v40/ (1 modele).
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v40 || exit 1
PY=$B/.venv/bin/python
$PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --w_ret 1 --ext_anchor $B/ext/coco_val_224.npy --full_classes 0,6,7,8,38 --teachers /p/scratch/training2666/huang25/Epi84/unlearning/v34/subs/rd_r2.pt,/p/scratch/training2666/huang25/Epi84/unlearning/v35/subs/rd40_r2.pt,/p/scratch/training2666/huang25/Epi84/unlearning/v38/subs/rdai_r2.pt --out subs/t3_5.pt > logs/t3_5.log 2>&1 || echo "ECHEC t3_5"
echo fini
