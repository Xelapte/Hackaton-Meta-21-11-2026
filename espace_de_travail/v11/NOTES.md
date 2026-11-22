# v11 — Cible moyennée sur la recette du record

## Pourquoi
- v9 : base « professeur cohérent » = 0.0833 ; la même base + cible moyennée sur 8 vues = 0.1242 (+50 %).
  La base v9 était mauvaise (extinction trop forte, comme b12), mais la cible moyennée semble aider.
- Jamais testée sur la recette du record (v5/f20_p8 + β7 = 0.1387).

## Idée
- Recette v5 : δ_full 20, δ_part, ancrage mosaïques sur l'original, puis extinction β=7 après coup.
- Seul changement : la base de la cible = moyenne des réponses de l'original sur 8 vues modifiées (aug 1.0) au lieu de
  l'image exacte (`--k_aug 8`). Le gold est une moyenne de plusieurs modèles → cible plus « moyennée ».
- Sur une base moyennée, la vraie classe domine moins → δ_part plus petit (6 et 4 au lieu de 8).

## Runs
| run | δ_part | acc entières | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| f20_k8_d6_b7 | 6 | 0.469 avant β | 0.925 | 0.82 avant β | **0.1462** (21/11 ~20h40, RECORD) |
| f20_k8_d4_b7 | 4 | 0.442 avant β | 0.975 | 0.83 avant β | |
