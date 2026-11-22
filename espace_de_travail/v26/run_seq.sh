#!/bin/bash
# v26 : 3 graines supplementaires de la recette du record (v18/k16_wr1_at025, graine 0), puis "soupe" = moyenne des
# poids des 4 graines, puis extinction beta=7. Sert aussi a mesurer le hasard d'un entrainement a l'autre.
# ~4 min par run. N'ecrit que dans v26/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v26 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
for S in 1 2 3; do
  echo "[$(date +%H:%M)] graine $S"
  $PY unlearn.py --target delta --tau 1.0 --delta_full 20 --delta_part 6 --aug 1.0 --k_aug 16 --w_ret 1 --aug_train 0.25 --seed $S \
     --out subs/s$S.pt > logs/s$S.log 2>&1 || echo "  -> ECHEC graine $S"
done
$PY head_shift.py subs/s1.pt 7 subs/s1_b7.pt >> logs/s1.log 2>&1
$PY soupe.py subs/soupe4.pt $B/v18/subs/k16_wr1_at025.pt subs/s1.pt subs/s2.pt subs/s3.pt > logs/soupe.log 2>&1 \
  && $PY head_shift.py subs/soupe4.pt 7 subs/soupe4_b7.pt >> logs/soupe.log 2>&1 || echo "  -> ECHEC soupe"
echo "[$(date +%H:%M)] fini"
