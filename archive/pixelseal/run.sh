#!/bin/bash
# Baseline Pixel Seal (fiche 6). Déposé dans la file du GPU 2 :
#   bash slurm/soumettre.sh 2 base bash pixelseal/run.sh   (secours : slurm/vague.sh)
# Variable attendue : IMAGES (dossier d'images de test).
# Le script se place à la racine du dépôt, là où se trouve le lien ckpts.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${IMAGES:?IMAGES non défini : dossier des images de test}"
mkdir -p logs
[ -e ckpts ] || { echo "Lien ckpts absent à la racine du dépôt (fiche 6)"; exit 1; }
echo "Pixel Seal sur CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-non défini}"

# Trois forces de tatouage en parallèle : le modèle est petit, le GPU a la place
for w in 0.2 0.3 0.4; do
    python pixelseal/quickstart.py --images "$IMAGES" --max_images 50 --scaling_w "$w" \
        > "logs/pixelseal_w$w.log" 2>&1 &
done
wait
grep -h "Score" logs/pixelseal_w*.log
