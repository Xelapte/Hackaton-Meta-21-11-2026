#!/bin/bash
# Pour le TITULAIRE DU NŒUD seulement : Slurm ne laisse entrer dans un job que la personne qui
# l'a lancé. Les autres membres passent par slurm/soumettre.sh (file d'attente).
# Ouvre un terminal sur le nœud réservé, limité à un seul GPU (non testé : à valider à 12h).
#   bash slurm/shell.sh <jobid> <gpu>        ex. : bash slurm/shell.sh 1234567 1
# <jobid> : numéro de la réservation du titulaire du nœud (squeue -A <projet>).
# Dans ce terminal, le GPU choisi s'appelle cuda:0 pour PyTorch et vLLM.
set -euo pipefail
JOB="${1:?usage : bash slurm/shell.sh <jobid> <gpu>}"
GPU="${2:?usage : bash slurm/shell.sh <jobid> <gpu>}"
case "$GPU" in 0|1|2|3) ;; *) echo "GPU doit valoir 0, 1, 2 ou 3"; exit 1;; esac
# Sans -c, srun ne donnerait qu'un cœur au terminal : on demande tous ceux du job,
# partagés avec les autres terminaux grâce à --overlap.
NCPU="$(squeue -j "$JOB" -h -o %C 2>/dev/null || true)"
exec srun --jobid="$JOB" --overlap --gres=gpu:4 ${NCPU:+-c "$NCPU"} --pty \
    env CUDA_VISIBLE_DEVICES="$GPU" PS1="[gpu$GPU] \w\$ " bash --norc
