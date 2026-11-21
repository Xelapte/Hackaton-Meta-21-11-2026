#!/bin/bash
# Le job du titulaire du nœud : il garde le nœud et exécute les files d'attente,
# une par GPU. Chaque membre y dépose ses commandes avec slurm/soumettre.sh, depuis
# son propre compte. Depuis la racine du clone du titulaire :
#   sbatch slurm/noeud.sh
# Pour la relève (sans payer deux nœuds) : bash slurm/relancer.sh
# Arrêt propre (après les commandes en cours) : touch <dossier_projet>/queue/STOP
#SBATCH --job-name=noeud
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=logs/noeud-%j.out
#SBATCH --reservation=cispahack      # réservation du hackathon (jusqu'au 22/11 14h)

source slurm/env.sh || exit 1
[ -d "$DIR/queue/gpu0/attente" ] || { echo "Lancez d'abord setup/partage.sh"; exit 1; }
rm -f "$DIR/queue/STOP"
echo "Nœud $(hostname), job $SLURM_JOB_ID : files d'attente ouvertes"
for g in 0 1 2 3; do
    bash slurm/worker.sh "$g" > "$DIR/queue/logs/worker-gpu$g-$SLURM_JOB_ID.log" 2>&1 &
done
wait
