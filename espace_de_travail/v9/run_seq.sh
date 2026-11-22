#!/bin/bash
# v9 "professeur coherent" : 5 entrainements (~2-3 min chacun) + enregistrement des reponses pour le simulateur (~1 min).
# A deposer dans la file d'un GPU de l'equipe. N'ecrit que dans v9/.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v9 || exit 1
mkdir -p subs logs cache
PY=$B/.venv/bin/python
run () {  # nom, puis arguments
  N=$1; shift
  echo "[$(date +%H:%M)] $N : $*"
  $PY unlearn.py "$@" --out subs/$N.pt > logs/$N.log 2>&1 || echo "  -> ECHEC $N (voir logs/$N.log)"
}
C="--target delta --tau 1.0 --teacher_beta 7 --init_shift"
# NB : 1er essai (delta_full 8) annule apres 1 min : les classes entieres restaient a 38 % dans la cible, car leur
# 2e choix est souvent une autre classe entiere. -> delta_full 20 (comme en v5), cible ~0 %.
run c7_f20_p8      $C --k_aug 0 --delta_full 20 --delta_part 8                    # base : professeur coherent
run c7_f20_p8_s8   $C --k_aug 0 --delta_full 20 --delta_part 8 --delta_slope 8    # + oubli proportionnel a la part retiree
run c7_f20_p8_a10  $C --k_aug 0 --delta_full 20 --delta_part 8 --aug_train 1.0    # + copies d'entrainement plus modifiees
run c7_f20_k8_d4   $C --k_aug 8 --aug 1.0 --delta_full 20 --delta_part 4          # + cible moyennee sur 8 vues
run c7_f20_k8_d6   $C --k_aug 8 --aug 1.0 --delta_full 20 --delta_part 6          # idem, oubli un peu plus fort
echo "[$(date +%H:%M)] cache"
A=""; for n in c7_f20_p8 c7_f20_p8_s8 c7_f20_p8_a10 c7_f20_k8_d4 c7_f20_k8_d6; do [ -f subs/$n.pt ] && A="$A $n=$B/v9/subs/$n.pt"; done
$PY cache.py $A > logs/cache.log 2>&1 || echo "  -> ECHEC cache"
echo "[$(date +%H:%M)] fini"
