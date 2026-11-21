#!/bin/bash
# Installe la bibliothèque videoseal (qui contient Pixel Seal) dans l'environnement
# de l'équipe. À lancer sur le NŒUD DE LOGIN, après download_all.sh.
# Usage : bash setup/install_pixelseal.sh <dossier_projet>
set -euo pipefail
DIR=${1:?"usage : bash setup/install_pixelseal.sh <dossier_projet>"}
source "$DIR/env/bin/activate"
cd "$DIR/repos/videoseal"

# requirements.txt ne contient pas decord, dont l'installation échoue parfois ;
# on installe donc le paquet lui-même sans ses dépendances.
pip install -r requirements.txt
pip install -e . --no-deps

python -c "import videoseal; print('videoseal', videoseal.__version__, 'importé')"
echo "Pensez au lien symbolique : ln -s $DIR/ckpts <racine_du_depot>/ckpts"
