# v14 — Masque de saillance à la SalUn (référence officielle de l'épreuve)

## Pourquoi
- Record : v11/f20_k8_d6_b7 = **0.1462** (cible moyennée sur 8 vues). Jusqu'ici on modifie TOUS les poids du modèle,
  ce qui le fait dériver de l'original (note retain, seuil des 70 %).
- SalUn (Fan et al., ICLR 2024) : ne modifier que les poids « saillants » pour les images à oublier.

## Idée
- Avant l'entraînement : pour chaque poids, |gradient| de la loss d'oubli cumulé sur les 3 500 images.
- Seule la fraction F des poids les plus saillants peut bouger (`--salun_frac F`) ; les autres restent exactement
  ceux de l'original. Le reste de la recette = record (k_aug 8, δ_full 20, δ_part 6, puis extinction β=7).
- Différence avec SalUn : eux réétiquettent au hasard ; nous gardons nos cibles (imiter le gold, pas le hasard).

## Runs
| run | fraction de poids libres | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| f20_k8_d6_sal0.5_b7 | 50 % | 0.477 | 0.925 | 0.81 (KL 0.098) | |
| f20_k8_d6_sal0.2_b7 | 20 % | 0.544 | 0.931 | 0.82 (KL 0.098) | |

Référence sans masque (v11/f20_k8_d6) : 0.469 / 0.925 / 0.82 (KL 0.095).
Lecture locale : le masque ne réduit PAS la dérive sur les mosaïques (KL 0.098 contre 0.095) et l'oubli est le même.
Les poids les plus saillants suffisent à faire tout le changement → pas de gain attendu, à confirmer par 1 soumission au plus.
