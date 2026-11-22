#!/bin/bash
# v31-a : groupe C = {29,30,31,33} traite comme partiel ; puis B et C tous deux partiels. ~4 min par run. N'ecrit que dans v31/.
source /p/scratch/training2666/huang25/Epi84/unlearning/v31/common.sh
run Cpart   "0,6,7,8,38"
run BCpart  "0,38"
echo "[$(date +%H:%M)] fini"
