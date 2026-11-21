#!/bin/bash
# Dépose une commande dans la file d'un GPU. Depuis la racine de VOTRE clone,
# sur le nœud de login :
#   bash slurm/soumettre.sh <gpu> <nom_court> <commande...>
#   ex. : bash slurm/soumettre.sh 1 scrub_base bash scrub/run.sh
#         bash slurm/soumettre.sh 0 smoke python met/generate_vllm.py --model "$MODEL_PATH" \
#               --prompts met/prompts_exemple.jsonl --n 3 --max_tokens 20 --seed 1 --out results/met/s.npz
# La commande s'exécute dans votre clone, avec slurm/env.sh chargé et CUDA_VISIBLE_DEVICES
# fixé. Les fichiers qu'elle lance sont lus au démarrage, pas au dépôt.
# Petit test (< 30 min, votre GPU) : libre. Le reste : accord du coordinateur du nœud.
set -euo pipefail
usage="usage : bash slurm/soumettre.sh <gpu> <nom_court> <commande...>"
GPU="${1:?$usage}"; NOM="${2:?$usage}"; shift 2
[ $# -gt 0 ] || { echo "$usage"; exit 1; }
case "$GPU" in 0|1|2|3) ;; *) echo "GPU doit valoir 0, 1, 2 ou 3"; exit 1;; esac
[ -f slurm/env.sh ] || { echo "Lancez depuis la racine de votre clone"; exit 1; }
source slurm/env.sh || exit 1
Q="$DIR/queue/gpu$GPU"
[ -d "$Q/attente" ] || { echo "Files absentes : le coordinateur doit lancer setup/partage.sh"; exit 1; }
ID="$(date +%Y%m%d-%H%M%S-%N)_${USER:-$(id -un)}_${NOM//[^A-Za-z0-9_.-]/_}"
TMP="$Q/attente/.$ID.tmp"
{
    echo "#!/bin/bash"
    echo "# Déposé par ${USER:-$(id -un)} le $(date '+%F %T')"
    printf 'cd %q || exit 97\n' "$PWD"
    echo "source slurm/env.sh || exit 98"
    printf '%q ' "$@"; echo
} > "$TMP"
mv "$TMP" "$Q/attente/$ID.sh"                    # dépôt atomique : jamais lu à moitié
N=$(find "$Q/attente" -maxdepth 1 -name '*.sh' | wc -l)
echo "Déposé sur le GPU $GPU : $ID ($N en attente, celle-ci comprise)"
echo "Journal : tail -f $DIR/queue/logs/$ID.log"
VIV=$(cat "$Q/vivant" 2>/dev/null || echo 0)
if [ $(( $(date +%s) - VIV )) -gt 60 ]; then
    echo "Attention : aucune file active sur ce GPU (le job slurm/noeud.sh tourne-t-il ?)"
fi
