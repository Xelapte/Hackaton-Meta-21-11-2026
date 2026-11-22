#!/bin/bash
# Telecharge UNE archive publique d'images dans unlearning/ext/ (substitut de retain, autorise par les organisateurs).
# Garde-fous : https/http uniquement, un seul fichier, jamais de decompression (quota de nombre de fichiers),
# rien n'est envoye (simple telechargement), journal dans ext/telechargements.txt.
# Usage : bash telecharge.sh <url> <nom_de_fichier>
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
URL="${1:?usage : bash telecharge.sh <url> <nom_de_fichier>}"; NOM="${2:?nom de fichier manquant}"
case "$URL" in http://*|https://*) ;; *) echo "refuse : URL http(s) uniquement"; exit 2;; esac
case "$NOM" in */*|.*) echo "refuse : nom de fichier simple uniquement"; exit 2;; esac
mkdir -p $B/ext && cd $B/ext || exit 1
[ -e "$NOM" ] && { echo "existe deja : ext/$NOM"; ls -la "$NOM"; exit 0; }
echo "[$(date '+%F %T')] $URL -> ext/$NOM" >> telechargements.txt
wget -q --show-progress --tries=3 --timeout=30 -O "$NOM.part" "$URL" && mv "$NOM.part" "$NOM" && ls -la "$NOM" || { echo "ECHEC du telechargement"; rm -f "$NOM.part"; exit 1; }
