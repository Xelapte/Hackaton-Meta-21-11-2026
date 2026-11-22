# v17 — Plus de vues dans la cible moyennée, copies d'entraînement plus légères

## Pourquoi
- Cible moyennée : 8 vues → 0.1462 ; 16 vues → **0.1506** (record, v15). La tendance monte : on teste 32 et 64.
- Copies d'entraînement plus fortes (aug_train 1.0) → 0.1307 contre 0.1412 : les copies cachées du serveur sont
  sans doute légères → on teste des copies plus légères (aug_train 0.25 au lieu de 0.5).

## Runs (recette du record : δ_full 20, δ_part 6, aug 1.0, puis β=7 ; déposé par Claude, autorisé par E.)
| run | changement | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| f20_k32_d6_b7 | 32 vues | 0.452 | 0.934 | 0.81 | |
| f20_k16_d6_at025_b7 | 16 vues + copies d'entraînement légères | 0.436 | 0.926 | 0.80 | **0.1526** (23h00, ≈ base 0.1506 : pas d effet net) |
| f20_k64_d6_b7 | 64 vues | 0.453 | 0.935 | 0.82 | **0.1523** (22h29, RECORD, soumis par Claude) |
