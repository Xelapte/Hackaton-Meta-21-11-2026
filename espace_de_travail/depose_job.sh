#!/bin/bash
# Depose UN script de version dans la file GPU de l'equipe (slurm/soumettre.sh, jamais de sbatch).
# Garde-fous : GPU 0-3, et le script doit etre un fichier unlearning/vN/xxx.sh existant.
# Usage : bash depose_job.sh <gpu> <nom_court> vN/run_seq.sh
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
GPU="${1:?usage : bash depose_job.sh <gpu> <nom_court> vN/script.sh}"; NOM="${2:?nom court manquant}"; S="${3:?script manquant}"
case "$GPU" in 0|1|2|3) ;; *) echo "refuse : GPU doit valoir 0, 1, 2 ou 3"; exit 2;; esac
case "$S" in v[0-9]*/*.sh) ;; *) echo "refuse : le script doit etre de la forme vN/xxx.sh"; exit 2;; esac
[ -f "$B/$S" ] || { echo "introuvable : $B/$S"; exit 2; }
cd /p/scratch/training2666/huang25/Epi84/membres/LOGIN || exit 1
echo "[$(date '+%F %T')] GPU $GPU $NOM $S" >> $B/jobs_journal.txt
bash slurm/soumettre.sh "$GPU" "$NOM" bash "$B/$S"
