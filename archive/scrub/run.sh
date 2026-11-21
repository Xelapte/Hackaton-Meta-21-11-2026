#!/bin/bash
# Baseline SCRUB (fiche 8). Déposé dans la file du GPU 1 :
#   bash slurm/soumettre.sh 1 base bash scrub/run.sh   (secours : slurm/vague.sh)
# Variables attendues : DIR (dossier projet, qui contient data/), facultatif OUT.
# Le modèle original et le réentraînement tournent en parallèle sur le même GPU :
# un ResNet-18 CIFAR n'utilise qu'une petite partie des 40 Go.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DIR:?DIR non défini (dossier projet)}"
D="$DIR/data"
O="${OUT:-results/scrub}"
mkdir -p "$O" logs
echo "SCRUB sur CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-non défini}"

# 1) et 2) en parallèle : original sur tout CIFAR-10, réentraînement sans le forget set
[ -f "$O/original.pt" ] || python scrub/train_original.py --data "$D" --epochs 30 \
    --out "$O/original.pt" > logs/scrub_original.log 2>&1 &
[ -f "$O/retrain_class.pt" ] || python scrub/train_original.py --data "$D" --epochs 30 \
    --retrain --mode class --out "$O/retrain_class.pt" > logs/scrub_retrain.log 2>&1 &
wait
[ -f "$O/original.pt" ] && [ -f "$O/retrain_class.pt" ] || {
    echo "Entraînement échoué : voir logs/scrub_original.log et logs/scrub_retrain.log"; exit 1; }

# 3) SCRUB et SCRUB+R, oubli de la classe 5 (réglages du tableau 3 du papier)
python scrub/run_scrub.py --data "$D" --original "$O/original.pt" \
    --retrain_ckpt "$O/retrain_class.pt" --mode class \
    --max_steps 2 --n_epochs 3 --bs_forget 512 --bs_retain 128
# 4) Variante : oubli sélectif de 100 exemples
python scrub/run_scrub.py --data "$D" --original "$O/original.pt" \
    --mode selective --n_forget 100 --max_steps 5 --n_epochs 5 \
    --bs_forget 16 --bs_retain 64
