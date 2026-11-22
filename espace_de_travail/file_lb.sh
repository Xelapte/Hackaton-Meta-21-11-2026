#!/bin/bash
# Ecoule la file de soumissions lb_file.txt (un fichier par ligne, le premier d'abord) via soumettre_lb.sh.
# Tourne detache (survit a la fermeture de la session). Pour changer l'ordre : editer lb_file.txt.
# Pour l'arreter : creer le fichier lb_STOP. Une seule instance a la fois (verrou lb_file.lock).
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B || exit 1
exec 9> $B/lb_file.lock; flock -n 9 || { echo "deja en cours"; exit 0; }
while [ ! -f $B/lb_STOP ]; do
  F=$(grep -m1 -v '^\s*$' $B/lb_file.txt 2>/dev/null)
  [ -n "$F" ] || { sleep 60; continue; }
  bash $B/soumettre_lb.sh "$F"; s=$?
  if [ $s -eq 3 ]; then sleep 30; continue; fi          # trop tot / serveur non joint : on reessaie
  grep -v -x -F "$F" $B/lb_file.txt > $B/lb_file.tmp; cat $B/lb_file.tmp > $B/lb_file.txt   # fait (ou refuse) : on retire la ligne
done
