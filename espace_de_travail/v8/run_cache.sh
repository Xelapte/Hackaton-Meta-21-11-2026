#!/bin/bash
# v8 : enregistre les reponses de tous nos modeles (aucun entrainement), ~4-5 min sur 1 GPU.
# A deposer dans la file d'un GPU de l'equipe. N'ecrit que dans v8/cache et v8/logs.
set -u
B=/p/scratch/training2666/huang25/Epi84/unlearning
cd $B/v8 || exit 1
mkdir -p cache logs
$B/.venv/bin/python cache.py \
  k1_a10=$B/v3/subs/k1_a10.pt t2=$B/v2/subs/t2.pt \
  d4=$B/v4/subs/d4.pt d8=$B/v4/subs/d8.pt d12=$B/v4/subs/d12.pt d16=$B/v4/subs/d16.pt \
  f20_p8=$B/v5/subs/f20_p8.pt f20_p10=$B/v5/subs/f20_p10.pt f20_p8_wr1=$B/v5/subs/f20_p8_wr1.pt \
  r_b8=$B/v7/subs/r_b8.pt r_b8_e5=$B/v7/subs/r_b8_e5.pt r_b8_wr6=$B/v7/subs/r_b8_wr6.pt r_b12=$B/v7/subs/r_b12.pt \
  > logs/cache.log 2>&1 || echo "ECHEC cache (voir logs/cache.log)"
echo "fini"
