#!/bin/bash
# Test de fumée sur un nœud GPU (10 minutes). À lancer depuis la racine du dépôt :
#   sbatch slurm/test.sh
# Remplissez d'abord slurm/env.sh. JURECA-DC : partition dc-gpu, réservation cispahack.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/test-%j.out
#SBATCH --reservation=cispahack      # réservation du hackathon (jusqu'au 22/11 14h)

source slurm/env.sh || exit 1
nvidia-smi
nvidia-smi topo -m                    # NV# = GPU reliés par NVLink
python common/smoke_test.py
# Le nœud de calcul voit-il et modifie-t-il le dossier projet ? (lu par setup/verif_partage.sh)
ls "$DIR" >/dev/null && touch "$DIR/queue/.calcul_$(hostname)" && echo "Dossier projet : visible et modifiable depuis $(hostname)"
