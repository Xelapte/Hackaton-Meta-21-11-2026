#!/bin/bash
# État des files d'attente des 4 GPU. Depuis la racine de votre clone :
#   bash slurm/file.sh
source slurm/env.sh >/dev/null || exit 1
for g in 0 1 2 3; do
    Q="$DIR/queue/gpu$g"
    VIV=$(cat "$Q/vivant" 2>/dev/null || echo 0); AGE=$(( $(date +%s) - VIV ))
    if [ "$AGE" -gt 60 ]; then ETAT="file INACTIVE"; else ETAT="file active"; fi
    echo "== GPU $g ($ETAT)"
    for f in "$Q"/en_cours/*.sh; do
        [ -e "$f" ] && echo "  en cours   : $(basename "$f" .sh)"
    done
    find "$Q/attente" -maxdepth 1 -name '*.sh' 2>/dev/null | sort | while read -r f; do
        echo "  en attente : $(basename "$f" .sh)"
    done
    find "$Q/fini" -maxdepth 1 -name '*.sh' -printf '%T@ %f\n' 2>/dev/null | sort -rn | \
        head -n 3 | while read -r _ f; do echo "  fini       : ${f%.sh}"; done
done
[ -e "$DIR/queue/STOP" ] && echo "STOP présent : les files s'arrêtent après les commandes en cours."
echo "Journaux : $DIR/queue/logs/"
