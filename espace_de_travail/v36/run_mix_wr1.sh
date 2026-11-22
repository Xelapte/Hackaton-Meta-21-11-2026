#!/bin/bash
# v36 : recette du record avec ancrage sur photos exterieures, variante mix_wr1 (--w_ret 1 --ext_frac 0.5). ~5 min. N'ecrit que dans v36/.
B=/p/scratch/training2666/huang25/Epi84/unlearning; cd $B/v36 || exit 1
PY=$B/.venv/bin/python
$PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --aug_train 0.25 --ext_anchor /p/scratch/training2666/huang25/Epi84/unlearning/ext/coco_val_224.npy --w_ret 1 --ext_frac 0.5 --out subs/mix_wr1.pt > logs/mix_wr1.log 2>&1 \
  && $PY head_shift.py subs/mix_wr1.pt 7 subs/mix_wr1_b7.pt 0,6,7,8,38 >> logs/mix_wr1.log 2>&1 || echo "ECHEC mix_wr1"
echo fini
