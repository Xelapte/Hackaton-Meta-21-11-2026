#!/bin/bash
# Arrête la commande EN COURS sur un GPU (la file passe à la suivante).
#   bash slurm/annuler.sh <gpu>
# Votre propre petit test : libre. La commande de quelqu'un d'autre : accord du
# coordinateur du nœud. Pour retirer une commande encore en attente :
#   rm <dossier_projet>/queue/gpu<gpu>/attente/<nom>.sh
set -euo pipefail
GPU="${1:?usage : bash slurm/annuler.sh <gpu>}"
case "$GPU" in 0|1|2|3) ;; *) echo "GPU doit valoir 0, 1, 2 ou 3"; exit 1;; esac
source slurm/env.sh >/dev/null || exit 1
Q="$DIR/queue/gpu$GPU"
EN=$(ls "$Q"/en_cours/*.sh 2>/dev/null || true)
[ -n "$EN" ] || { echo "Rien ne tourne sur le GPU $GPU"; exit 0; }
echo "Annulation demandée : $(basename "$EN" .sh)"
touch "$Q/ANNULER"
