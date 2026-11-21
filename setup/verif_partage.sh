#!/bin/bash
# Vérifie que vous pouvez travailler dans le dossier projet partagé. Chaque membre,
# sur le nœud de login, depuis la racine de son clone :  bash setup/verif_partage.sh
source slurm/env.sh >/dev/null || exit 1
ok=1
echo "Vous : $(id -un) | groupes : $(id -Gn)"
echo "Nœud de login : $(hostname)   (notez-le : vos sessions tmux restent sur ce nœud)"
if [ "$(stat -c %U "$DIR")" != "$(id -un)" ] && ! getfacl -p "$DIR" 2>/dev/null | grep -q "^user:$(id -un):rw"; then
    echo "ÉCHEC : le dossier de l'équipe ne vous est pas ouvert (setup/partage.sh avec votre identifiant)"; ok=0
fi
case "$(stat -c %A "$DIR")" in
    ???????---) ;;
    *) echo "ATTENTION : le dossier est lisible par d'autres que l'équipe : relancez setup/partage.sh"; ok=0;;
esac
for g in 0 1 2 3; do
    f="$DIR/queue/gpu$g/attente/.verif_$(id -un)"
    if touch "$f" 2>/dev/null; then rm -f "$f"; else echo "ÉCHEC : écriture impossible dans la file du GPU $g"; ok=0; fi
done
touch "$DIR/scores.csv" 2>/dev/null || { echo "ÉCHEC : journal des scores non modifiable"; ok=0; }
grep -qs "umask 002" ~/.bashrc || { echo "ATTENTION : ajoutez « umask 002 » à ~/.bashrc"; ok=0; }
for c in "$DIR"/membres/*/; do
    [ -r "$c" ] && [ -x "$c" ] || { echo "ATTENTION : clone illisible : $c"; ok=0; }
done
ls "$DIR"/queue/.calcul_* >/dev/null 2>&1 && echo "Le nœud de calcul a déjà écrit dans le dossier : $(ls "$DIR"/queue/.calcul_* | xargs -n1 basename | tr '\n' ' ')"
[ $ok = 1 ] && echo "PARTAGE OK" || echo "PARTAGE INCOMPLET : voir fiche 4, section 4, et prévenir le titulaire"
