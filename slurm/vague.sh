#!/bin/bash
# SECOURS si les files d'attente (slurm/noeud.sh) ne marchent pas.
# Une « vague » : un processus par GPU, en parallèle. Depuis la racine du dépôt :
#   sbatch slurm/vague.sh
# Remplissez d'abord slurm/env.sh et les valeurs entre chevrons ci-dessous.
# Chaque tâche lance son <tache>/run.sh : modifiez ces fichiers pour changer d'expérience.
#SBATCH --account=<projet>
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=logs/vague-%j.out
##SBATCH --reservation=<nom>        # si les organisateurs en fournissent une

source slurm/env.sh || exit 1
CUDA_VISIBLE_DEVICES=0 bash met/run.sh        > "logs/gpu0-$SLURM_JOB_ID.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash scrub/run.sh      > "logs/gpu1-$SLURM_JOB_ID.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash pixelseal/run.sh  > "logs/gpu2-$SLURM_JOB_ID.log" 2>&1 &
# GPU 3 (joker) : une copie modifiée d'un run.sh, avec un autre dossier de sortie (OUT=...).
# Vide = GPU 3 libre pour la démo.
JOKER=""                   # ex. : JOKER="met/run_joker.sh"
if [ -n "$JOKER" ]; then
    CUDA_VISIBLE_DEVICES=3 bash "$JOKER"      > "logs/gpu3-$SLURM_JOB_ID.log" 2>&1 &
fi
wait
