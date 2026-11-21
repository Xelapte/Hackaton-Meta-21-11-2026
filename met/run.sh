#!/bin/bash
# Baseline Model Equality Testing (fiche 7). Déposé dans la file du GPU 0 :
#   bash slurm/soumettre.sh 0 base bash met/run.sh   (secours : slurm/vague.sh)
# Variables attendues : MODEL_PATH (dossier du modèle), et facultatif PROMPTS, OUT.
# Pour une deuxième instance (GPU joker), copiez ce fichier et changez OUT.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?MODEL_PATH non défini (dossier du modèle téléchargé)}"
[ -d "$MODEL_PATH" ] || { echo "Modèle introuvable : $MODEL_PATH"; exit 1; }
P="${PROMPTS:-met/prompts_exemple.jsonl}"
O="${OUT:-results/met}"
mkdir -p "$O"
echo "MET sur CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-non défini}"

# 1) Référence P et contrôle P' : même réglage, graines différentes
python met/generate_vllm.py --model "$MODEL_PATH" --prompts "$P" --dtype float32 --seed 1 --out "$O/P.npz"
python met/generate_vllm.py --model "$MODEL_PATH" --prompts "$P" --dtype float32 --seed 2 --out "$O/P2.npz"
# 2) Variantes Q, une graine différente chacune
for v in nf4:3 int8:4 watermark:5; do
    python met/generate_hf.py --model "$MODEL_PATH" --prompts "$P" --variant "${v%:*}" \
        --seed "${v#*:}" --out "$O/Q_${v%:*}.npz"
done
# 3) Comparaisons (chaque appel écrit une ligne dans le journal des scores)
python met/compare.py "$O/P.npz" "$O/P2.npz" --name controle
for v in nf4 int8 watermark; do
    python met/compare.py "$O/P.npz" "$O/Q_$v.npz" --name "$v"
done
