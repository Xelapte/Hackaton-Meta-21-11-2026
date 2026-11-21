#!/bin/bash
# Variables communes aux jobs (test.sh, noeud.sh, vague.sh) et à chaque commande des files.
# À remplir une fois, puis depuis la racine du dépôt :  source slurm/env.sh
export DIR="/p/project1/training2666/team_Epi84"
case "$DIR" in "<"*) echo "Remplacez <dossier_projet> dans slurm/env.sh"
                     return 1 2>/dev/null || exit 1;; esac
# Modules JSC : le même Python sur le login et sur les nœuds GPU (vérifier : module spider Python)
export MODULES="Stages/2025 GCC Python"
module load $MODULES 2>/dev/null || echo "Attention : module load $MODULES a échoué (voir module spider Python)"
[ -f "$DIR/env/bin/activate" ] && source "$DIR/env/bin/activate"
export HF_HOME="$DIR/hf_cache"
export HF_HUB_OFFLINE=1                           # pas d'internet sur les nœuds de calcul
export SCORES_CSV="$DIR/scores.csv"               # journal des scores commun à toute l'équipe
umask 002                                         # fichiers lisibles et modifiables par l'équipe
export MODEL_PATH="$DIR/models/<nom_du_modele>"   # Model Equality Testing, à remplacer
export IMAGES="<dossier_images>"                  # Pixel Seal : images de test (connu à 12h)
