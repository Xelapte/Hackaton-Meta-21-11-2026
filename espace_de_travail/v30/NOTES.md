# v30 — Le groupe {6, 7, 8} traité comme des classes partielles

## Pourquoi
- v27/sans_B = **0.1609** (record) : en n'éteignant PAS les classes 6, 7, 8, le score monte de 3,7 %.
- v25/mfa (empêcher de ranger une image d'une classe « entière » dans une autre classe « entière ») = 0.1169 : perd.
- Les deux vont dans le même sens : le gold connaît probablement les classes 6, 7, 8. Elles ont 129–130 images dans le
  forget set, mais 130 serait un plafond, pas la taille de la classe.
- Dans sans_B, ces classes restent entraînées avec la cible « classe entière » (δ 20, ≈ 32 % de bonnes réponses).
  Si le gold les connaît, il faut les traiter comme les autres classes partielles (≈ 90 % de bonnes réponses).

## Runs (recette du record ; classes entières = {0, 29, 30, 31, 33, 38} ; β=7 sur ces 6 classes)
| run | δ_part | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| Bpart_d6_b7 | 6 | | | | |
| Bpart_d3_b7 | 3 | | | | |
