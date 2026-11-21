#!/bin/bash
# Exécute l'une après l'autre les commandes déposées dans la file d'un GPU.
# Lancé par slurm/noeud.sh (un par GPU) : ne pas lancer à la main.
#   attente/  commandes déposées, exécutées dans l'ordre de dépôt
#   en_cours/ la commande qui tourne
#   fini/     commandes terminées, avec leur code de sortie dans le nom
set -uo pipefail
GPU="${1:?usage : worker.sh <gpu>}"
: "${DIR:?DIR non défini : lancer via slurm/noeud.sh}"
Q="$DIR/queue/gpu$GPU"; L="$DIR/queue/logs"
umask 002
echo "$(date '+%F %T') file du GPU $GPU ouverte sur $(hostname)"
# Une commande restée « en cours » vient d'un job précédent arrêté : on la classe
for f in "$Q"/en_cours/*.sh; do
    [ -e "$f" ] && mv "$f" "$Q/fini/$(basename "$f" .sh)__interrompue.sh"
done
while [ ! -e "$DIR/queue/STOP" ]; do
    date +%s > "$Q/vivant"                       # signe de vie, lu par slurm/file.sh
    F=$(find "$Q/attente" -maxdepth 1 -name '*.sh' 2>/dev/null | sort | head -n 1)
    if [ -z "$F" ]; then sleep 5; continue; fi
    ID=$(basename "$F" .sh)
    mv "$F" "$Q/en_cours/$ID.sh" 2>/dev/null || continue    # retirée entre-temps
    rm -f "$Q/ANNULER"
    echo "$(date '+%F %T') début sur le GPU $GPU ($(hostname))" > "$L/$ID.log"
    echo "$(date '+%F %T') début $ID"
    # setsid : la commande et tous ses enfants forment un groupe qu'on peut annuler
    CUDA_VISIBLE_DEVICES="$GPU" setsid bash "$Q/en_cours/$ID.sh" >> "$L/$ID.log" 2>&1 &
    PID=$!
    while kill -0 "$PID" 2>/dev/null; do
        date +%s > "$Q/vivant"
        if [ -e "$Q/ANNULER" ]; then
            echo "$(date '+%F %T') ANNULÉE" >> "$L/$ID.log"
            kill -TERM -- "-$PID" 2>/dev/null; sleep 10; kill -KILL -- "-$PID" 2>/dev/null
            rm -f "$Q/ANNULER"
        fi
        sleep 2
    done
    wait "$PID"; RC=$?
    echo "$(date '+%F %T') fin, code $RC" >> "$L/$ID.log"
    echo "$(date '+%F %T') fin $ID, code $RC"
    mv "$Q/en_cours/$ID.sh" "$Q/fini/${ID}__code$RC.sh"
done
echo "$(date '+%F %T') STOP : file du GPU $GPU fermée"
