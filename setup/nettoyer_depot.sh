#!/bin/bash
# Une fois, dans VOTRE clone, après avoir copié le nouveau kit par-dessus (cp -r) :
# retire les anciens fichiers déplacés dans archive/ (git rm), puis il reste à commiter.
#   bash setup/nettoyer_depot.sh && git commit -m "Ancien kit déplacé dans archive/" && git push
set -euo pipefail
[ -d archive ] || { echo "Lancez depuis la racine du clone, après avoir copié le nouveau kit"; exit 1; }
for p in setup/install_env.sh setup/download_all.sh setup/install_pixelseal.sh \
         met/generate_vllm.py met/generate_hf.py met/run.sh met/compare.py \
         met/prompts_exemple.jsonl met/CLAUDE.md scrub pixelseal slurm/vague.sh; do
    if [ -e "$p" ]; then git rm -r -q "$p" && echo "retiré : $p (copie dans archive/$p)"; fi
done
git add -A
git status --short | head -40
