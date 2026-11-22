# v13 — Idées jamais testées sur la bonne base, une à la fois

## Pourquoi
- v9 avait testé ces idées sur une base qui s'est révélée mauvaise (professeur cohérent : 0.0833) → rien appris.
- Base de référence ici : record v6/f20_p10_b7 = **0.1412** (recette v5 : δ_full 20, δ_part 10, puis extinction β=7).

## Runs (code = v10/unlearn.py, tilt et professeur désactivés ; chaque run change UN élément)
| run | changement | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| f20_p10_s8_b7 | oubli proportionnel à la part retirée (`--delta_slope 8`) | 0.468 | 0.873 | 0.80 | **0.1400** (21/11 ~21h, ≈ base 0.1412 : pas d effet) |
| f20_p10_a10_b7 | copies d'entraînement plus modifiées (`--aug_train 1.0`) | 0.529 | 0.901 | 0.80 | **0.1307** (21/11 ~21h10, < base 0.1412 : copies plus modifiées = moins bien) |
| f20_k8_d8_b7 | cible moyennée sur 8 vues, δ_part 8 | 0.509 | 0.846 | 0.80 | |
