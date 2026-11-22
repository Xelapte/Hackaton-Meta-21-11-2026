#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# A lancer DEPUIS le dossier de version :  cd unlearning/v1 && sbatch sweep.sh
# 4 runs en parallele, 1 par GPU. Modifie TAUS / EXTRA pour changer le sweep.
cd "$SLURM_SUBMIT_DIR" || exit 1
source ../.venv/bin/activate
mkdir -p subs logs

TAUS=(1.0 1.5 2.0 3.0)
EXTRA="--epochs 3 --lr 1e-5"

for i in 0 1 2 3; do
  T=${TAUS[$i]}
  CUDA_VISIBLE_DEVICES=$i python unlearn.py --tau $T $EXTRA \
    --out subs/tau${T}.pt > logs/tau${T}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
