#!/bin/bash
# Programme la relève du job du nœud : un nouveau noeud.sh qui démarrera dès que
# l'actuel se terminera (fin de --time ou STOP). Jamais deux nœuds payés en même temps,
# et les commandes en attente passent au job suivant. À lancer par le titulaire du nœud
# (ou le relais de nuit), sur le nœud de login, depuis la racine de son clone :
#   bash slurm/relancer.sh
# Lancez-le dès que le job tourne : la relève attend dans la file Slurm.
set -euo pipefail
ACTUEL=$(squeue -u "${USER:-$(id -un)}" -h -n noeud -o %i | sort -n | tail -n 1)
if [ -z "$ACTUEL" ]; then
    echo "Aucun job noeud.sh en cours ou en attente à votre nom : sbatch slurm/noeud.sh"
    exit 1
fi
sbatch --dependency=afterany:"$ACTUEL" slurm/noeud.sh
echo "Relève programmée : elle démarrera à la fin du job $ACTUEL."
echo "Si un autre membre porte le job, c'est à lui de programmer la relève."
