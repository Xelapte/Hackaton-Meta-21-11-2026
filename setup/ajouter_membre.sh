#!/bin/bash
# Ouvre le dossier de l'équipe à un membre ajouté APRÈS setup/partage.sh.
# Linux ne laisse modifier les droits d'un fichier qu'à son propriétaire : CHAQUE membre
# déjà présent lance cette commande une fois, pour ses propres fichiers.
#   bash setup/ajouter_membre.sh <dossier_projet> <identifiant_du_nouveau>
set -uo pipefail
DIR=${1:?"usage : bash setup/ajouter_membre.sh <dossier_projet> <identifiant>"}
U=${2:?"usage : bash setup/ajouter_membre.sh <dossier_projet> <identifiant>"}
id "$U" >/dev/null 2>&1 || { echo "Identifiant inconnu : $U"; exit 1; }
MOI=$(id -un)
# Fichiers et dossiers qui vous appartiennent : accès au nouveau membre
find "$DIR" -user "$MOI" -exec setfacl -m "u:$U:rwX" {} + 2>/dev/null
# Vos dossiers : le nouveau membre aura aussi accès à ce qui y sera créé ensuite
find "$DIR" -type d -user "$MOI" -exec setfacl -d -m "u:$U:rwX" {} + 2>/dev/null
setfacl -m "u:$U:x" "$(dirname "$DIR")" 2>/dev/null || true
N=$(find "$DIR" -user "$MOI" | wc -l)
echo "$U a maintenant accès à vos $N fichiers et dossiers dans $DIR."
echo "Les autres membres doivent lancer la même commande pour les leurs."
