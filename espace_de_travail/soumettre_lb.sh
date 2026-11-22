#!/bin/bash
# Soumet UN fichier au leaderboard via submission.py (inchange), avec la cle lue dans ~/.hack_api_key (jamais affichee).
# Garde-fous : le fichier doit exister sous unlearning/v*/subs/, et 10 min au moins depuis la soumission precedente
# faite par ce script. Chaque appel est journalise dans lb_journal.txt.
# Usage : bash soumettre_lb.sh v15/subs/f20_k16_d6_b7.pt
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B || exit 1
# Interrupteur general : si le fichier lb_STOP existe, plus AUCUN envoi automatique (code 2 = les boucles s'arretent).
[ -f $B/lb_STOP ] && { echo "arrete : lb_STOP present, aucun envoi"; exit 2; }
F="${1:?usage : bash soumettre_lb.sh vN/subs/fichier.pt}"
case "$F" in v[0-9]*/subs/*.pt) ;; *) echo "refuse : le fichier doit etre de la forme vN/subs/xxx.pt"; exit 2;; esac
[ -f "$F" ] || { echo "introuvable : $F"; exit 2; }
# Anti-doublon : un fichier deja note (liste lb_faits.txt) n'est jamais renvoye.
grep -q -x -F "$F" $B/lb_faits.txt 2>/dev/null && { echo "deja soumis : $F"; exit 2; }
T=$B/.lb_derniere
if [ -f $T ]; then
  D=$(( $(date +%s) - $(cat $T) ))
  # On attend 8 min apres la derniere notation, puis on tente (la boucle appelante reessaie toutes les 30 s).
  [ $D -ge 480 ] || { echo "trop tot : derniere soumission il y a ${D}s (attendre $((480 - D))s)"; exit 3; }
fi
date +%s > $T
echo "[$(date '+%F %T')] $F" >> $B/lb_journal.txt
OUT=$(TEAM_API_KEY=$(cat ~/.hack_api_key) SUBMISSION_FILE=$B/$F .venv/bin/python submission.py 2>&1)
echo "$OUT" | tee -a $B/lb_journal.txt
if echo "$OUT" | grep -q -E "Could not reach the server|HTTP 429"; then
  # Serveur non joint ou delai de 10 min pas fini (HTTP 429). La tentative ne compte pas :
  # nouvel essai autorise dans 30 s. Code 3 = "reessayer plus tard".
  echo $(( $(date +%s) - 450 )) > $T
  exit 3
fi
echo "$OUT" | grep -q "Score (public split)" && echo "$F" >> $B/lb_faits.txt
# Le serveur compte ses 10 min a partir de la FIN de la notation : on repart de maintenant.
date +%s > $T
