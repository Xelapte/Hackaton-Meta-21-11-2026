#!/bin/bash
# Prépare le dossier de l'équipe. À lancer UNE fois, par le titulaire du nœud, sur le
# nœud de login, avec les identifiants JuDoor de TOUS les membres de l'équipe :
#   bash setup/partage.sh <dossier_projet> <identifiant1> <identifiant2> ...
# Le groupe Unix du projet (ex. training2666) est commun à TOUTES les équipes : on ne
# s'en sert pas. Le dossier est fermé au groupe et ouvert aux seuls membres listés,
# par ACL (setfacl), y compris pour les fichiers créés plus tard (ACL par défaut).
# Membre ajouté plus tard : setup/ajouter_membre.sh (chaque membre, pour ses fichiers).
set -euo pipefail
DIR=${1:?"usage : bash setup/partage.sh <dossier_projet> <identifiant> [<identifiant>...]"}
shift
[ $# -gt 0 ] || { echo "Donnez les identifiants JuDoor des membres de l'équipe"; exit 1; }
MOI=$(id -un)
for g in 0 1 2 3; do
    mkdir -p "$DIR/queue/gpu$g/attente" "$DIR/queue/gpu$g/en_cours" "$DIR/queue/gpu$g/fini"
done
mkdir -p "$DIR/queue/logs" "$DIR/membres"
chmod -R go-rwx "$DIR" 2>/dev/null || true   # fermé aux autres équipes (vos fichiers)
for u in "$MOI" "$@"; do
    id "$u" >/dev/null 2>&1 || { echo "Identifiant inconnu : $u"; exit 1; }
    setfacl -R -m "u:$u:rwX" "$DIR" 2>/dev/null || true      # accès aux fichiers existants
    setfacl -R -d -m "u:$u:rwX" "$DIR" 2>/dev/null || true   # et à ceux créés ensuite
done
# Le dossier parent doit être traversable par les membres (cas d'un sous-dossier).
P=$(dirname "$DIR")
for u in "$@"; do setfacl -m "u:$u:x" "$P" 2>/dev/null || true; done
echo "Dossier de l'équipe : $DIR"
getfacl -p "$DIR" | grep -E "^(user|group|other|mask)"
echo "Chaque membre : bash setup/verif_partage.sh (doit finir par PARTAGE OK)"
