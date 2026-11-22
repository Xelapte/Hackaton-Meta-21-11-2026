#!/bin/bash
# v16 : exploration LARGE autour du record (v11/f20_k8_d6_b7 = 0.1462). 8 modeles varies, plusieurs reglages changent
# a la fois (ancrage, vitesse/duree, temperature, poids des copies, couches libres...). Puis extinction beta=7.
# ~20-25 min au total. N'ecrit que dans v16/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v16 || exit 1
mkdir -p subs logs
PY=$B/.venv/bin/python
run () {  # nom, puis arguments ; puis extinction beta=7
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py --target delta --delta_full 20 "$@" --out subs/$N.pt > logs/$N.log 2>&1 \
    && $PY head_shift.py subs/$N.pt 7 subs/${N}_b7.pt >> logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
# reference (record) : --k_aug 8 --aug 1.0 --delta_part 6 --tau 1.0 --w_ret 3 --lr 5e-5 --epochs 5 --w_aug 1 --aug_train 0.5 --train_blocks 0
run e1 --k_aug 8  --aug 1.0 --delta_part 6 --w_ret 1                                   # ancrage plus faible
run e2 --k_aug 8  --aug 1.0 --delta_part 6 --w_ret 6 --epochs 8                        # ancrage plus fort, plus long
run e3 --k_aug 8  --aug 1.0 --delta_part 4 --tau 1.5                                   # cible plus hesitante
run e4 --k_aug 8  --aug 1.0 --delta_part 6 --lr 1e-4 --epochs 3 --w_aug 2              # rapide, plus de poids aux copies
run e5 --k_aug 8  --aug 1.0 --delta_part 6 --train_blocks 4 --lr 1e-4 --epochs 8       # seuls les 4 derniers blocs bougent
run e6 --k_aug 16 --aug 1.1 --delta_part 5 --w_ret 2 --aug_train 0.75                  # plus de vues, copies plus modifiees
run e7 --k_aug 8  --aug 1.0 --delta_part 7 --lr 2e-5 --epochs 10 --w_aug 0.5           # lent et long
run e8 --k_aug 16 --aug 1.0 --delta_part 5 --tau 1.25 --w_ret 1 --train_blocks 6       # melange
echo "[$(date +%H:%M)] fini"
