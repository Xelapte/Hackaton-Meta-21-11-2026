#!/bin/bash
# Prépare le dossier projet pour une équipe de plusieurs comptes. À lancer UNE fois,
# par le coordinateur du nœud, sur le NŒUD DE LOGIN :
#   bash setup/partage.sh <dossier_projet>
# Crée les files d'attente des GPU et rend tout le dossier modifiable par le groupe
# du projet. Chaque membre doit aussi ajouter « umask 002 » à son ~/.bashrc.
set -euo pipefail
DIR=${1:?"usage : bash setup/partage.sh <dossier_projet>"}
for g in 0 1 2 3; do
    mkdir -p "$DIR/queue/gpu$g/attente" "$DIR/queue/gpu$g/en_cours" "$DIR/queue/gpu$g/fini"
done
mkdir -p "$DIR/queue/logs" "$DIR/membres"
chmod -R g+rwX "$DIR"
find "$DIR" -type d -exec chmod g+s {} +      # les nouveaux fichiers gardent le groupe
echo "Groupe du dossier : $(stat -c %G "$DIR")  (doit être le groupe du projet)"
echo "Chaque membre : echo 'umask 002' >> ~/.bashrc, puis clone le dépôt dans"
echo "  $DIR/membres/\$USER/"
