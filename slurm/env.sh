#!/bin/bash
# Variables communes aux jobs (test.sh, vague.sh) et aux terminaux interactifs.
# À remplir une fois, puis depuis la racine du dépôt :  source slurm/env.sh
export DIR="<dossier_projet>"                     # à remplacer
case "$DIR" in "<"*) echo "Remplacez <dossier_projet> dans slurm/env.sh"
                     return 1 2>/dev/null || exit 1;; esac
source "$DIR/env/bin/activate"
export HF_HOME="$DIR/hf_cache"
export HF_HUB_OFFLINE=1                           # pas d'internet sur les nœuds de calcul
export SCORES_CSV="$DIR/scores.csv"               # journal des scores commun à toute l'équipe
umask 002                                         # fichiers lisibles et modifiables par l'équipe
export MODEL_PATH="$DIR/models/<nom_du_modele>"   # Model Equality Testing, à remplacer
export IMAGES="<dossier_images>"                  # Pixel Seal : images de test (connu à 12h)
