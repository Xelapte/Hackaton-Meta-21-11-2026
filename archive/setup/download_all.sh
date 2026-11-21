#!/bin/bash
# À lancer sur le NŒUD DE LOGIN (il a internet ; les nœuds de calcul n'en ont pas),
# avec l'environnement de l'équipe activé.
# Usage : bash setup/download_all.sh <dossier_projet> [<modele_hugging_face>]
# Le modèle est facultatif : la tâche 2 réelle passe par une API (pas de modèle local).
# Les données des tâches viennent de hackathon_setup.sh (organisateurs), pas d'ici.
set -euo pipefail
DIR=${1:?"usage : bash setup/download_all.sh <dossier_projet> <modele_hf>"}
MODEL=${2:-}
export HF_HOME="$DIR/hf_cache"
mkdir -p "$DIR/data" "$DIR/repos" "$DIR/models" "$DIR/ckpts" "$HF_HOME"

echo "== 1/4 Dépôts de code"
[ -d "$DIR/repos/videoseal" ] || \
    git clone https://github.com/facebookresearch/videoseal "$DIR/repos/videoseal"
[ -d "$DIR/repos/model-equality-testing" ] || \
    git clone https://github.com/i-gao/model-equality-testing \
        "$DIR/repos/model-equality-testing"
[ -d "$DIR/repos/SCRUB" ] || \
    git clone https://github.com/Meghdad92/SCRUB "$DIR/repos/SCRUB"

echo "== 2/4 Poids de Pixel Seal (et VideoSeal 1.0 pour comparer)"
# videoseal cherche ses poids dans ./ckpts/<dossier>_<fichier> (dossier courant)
dl() { [ -f "$2" ] || wget -q --show-progress -O "$2" "$1"; }
dl https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth \
    "$DIR/ckpts/pixelseal_checkpoint.pth"
dl https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth \
    "$DIR/ckpts/videoseal_y_256b_img.pth"

echo "== 3/4 CIFAR-10 et CIFAR-100 (SCRUB)"
python - "$DIR/data" <<'EOF'
import sys, torchvision
for train in (True, False):
    torchvision.datasets.CIFAR10(sys.argv[1], train=train, download=True)
    torchvision.datasets.CIFAR100(sys.argv[1], train=train, download=True)
print("CIFAR ok")
EOF

if [ -n "$MODEL" ]; then
    echo "== 4/4 Modèle de langage : $MODEL"
    DEST="$DIR/models/$(basename "$MODEL")"
    hf download "$MODEL" --local-dir "$DEST" || \
        huggingface-cli download "$MODEL" --local-dir "$DEST"
else
    echo "== 4/4 Pas de modèle de langage demandé : étape sautée"
fi

echo "Terminé. Dans la racine du dépôt d'équipe : ln -s $DIR/ckpts ckpts"
du -sh "$DIR"/{data,models,repos,ckpts} 2>/dev/null || true
