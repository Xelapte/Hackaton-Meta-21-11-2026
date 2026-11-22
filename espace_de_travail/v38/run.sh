#!/bin/bash
# v38 : re-distillation sur 20 000 images GENEREES PAR IA (DiffusionDB) au lieu de photos reelles. Preparation (~4 min de
# CPU) puis 2 blocs remis a zero, 6000 pas (~11 min). N'ecrit que dans v38/ et ext/.
cd /p/scratch/training2666/huang25/Epi84/unlearning/v38 || exit 1
PY=/p/scratch/training2666/huang25/Epi84/unlearning/.venv/bin/python
[ -f /p/scratch/training2666/huang25/Epi84/unlearning/ext/ddb_20k_224.npy ] || $PY prep_multi.py /p/scratch/training2666/huang25/Epi84/unlearning/ext/ddb_20k_224.npy /p/scratch/training2666/huang25/Epi84/unlearning/ext/diffusiondb_0000*.zip > logs/prep.log 2>&1 || echo "ECHEC prep"
$PY redistill.py --reinit 2 --steps 6000 --data /p/scratch/training2666/huang25/Epi84/unlearning/ext/ddb_20k_224.npy --out /p/scratch/training2666/huang25/Epi84/unlearning/v38/subs/rdai_r2.pt > logs/rdai_r2.log 2>&1 || echo "ECHEC rdai_r2"
echo fini
