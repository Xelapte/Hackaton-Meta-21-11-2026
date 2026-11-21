#!/bin/bash
# Ferme un dossier aux autres équipes et l'ouvre aux seuls membres listés (ACL).
# Pour le dossier OFFICIEL créé par hackathon_setup.sh, s'il est lisible par tout le
# groupe training2666 (commun à toutes les équipes). Par le chef d'équipe :
#   bash setup/fermer.sh <dossier> <identifiant1> <identifiant2> ...
# Ne touche qu'aux fichiers qui vous appartiennent : chaque membre qui y a créé des
# fichiers relance la commande pour les siens.
set -uo pipefail
D=${1:?"usage : bash setup/fermer.sh <dossier> <identifiant> [...]"}; shift
[ $# -gt 0 ] || { echo "Donnez les identifiants des membres"; exit 1; }
[ -d "$D" ] || { echo "Dossier introuvable : $D"; exit 1; }
MOI=$(id -un)
find "$D" -user "$MOI" -exec chmod go-rwx {} + 2>/dev/null
for u in "$MOI" "$@"; do
    id "$u" >/dev/null 2>&1 || { echo "Identifiant inconnu : $u"; exit 1; }
    find "$D" -user "$MOI" -exec setfacl -m "u:$u:rwX" {} + 2>/dev/null
    find "$D" -type d -user "$MOI" -exec setfacl -d -m "u:$u:rwX" {} + 2>/dev/null
done
# Les dossiers parents doivent rester traversables par les membres
P=$(dirname "$D")
for u in "$@"; do setfacl -m "u:$u:x" "$P" 2>/dev/null || true; done
echo "Fermé aux autres équipes : $D"
getfacl -p "$D" | grep -E "^(user|group|other)"
