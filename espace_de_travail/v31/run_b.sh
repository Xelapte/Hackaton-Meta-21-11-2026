#!/bin/bash
# v31-b : autres combinaisons (A = {0} partiel, D = {38} partiel, aucune classe entiere). ~4 min par run. N'ecrit que dans v31/.
source /p/scratch/training2666/huang25/Epi84/unlearning/v31/common.sh
run ABpart  "29,30,31,33,38"
run BDpart  "0,29,30,31,33"
run aucune  ""
echo "[$(date +%H:%M)] fini"
