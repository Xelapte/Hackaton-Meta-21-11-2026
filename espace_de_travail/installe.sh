#!/bin/bash
# Installe des paquets Python dans un dossier A PART (unlearning/pylib_gen), jamais dans le .venv de l'equipe.
# Sert uniquement a faire tourner un generateur d'images (diffusers) pour fabriquer des images de remplacement.
# Utilise uv, en visant la meme version de Python que le .venv (3.12) pour que les paquets soient compatibles.
# Usage : bash installe.sh paquet1 paquet2 ...
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
[ $# -gt 0 ] || { echo "usage : bash installe.sh paquet1 paquet2 ..."; exit 2; }
for p in "$@"; do case "$p" in -*|*/*|*:*) echo "refuse : noms de paquets PyPI uniquement ($p)"; exit 2;; esac; done
mkdir -p $B/pylib_gen
echo "[$(date '+%F %T')] $*" >> $B/pylib_gen/installations.txt
uv pip install --python $B/.venv/bin/python --target $B/pylib_gen "$@" 2>&1 | tail -4
