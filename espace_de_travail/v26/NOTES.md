# v26 — Mesurer le hasard entre entraînements, et « soupe » de modèles

## Pourquoi
- v16/e1 (8 vues, ancrage 1) = 0.0848 alors que v18/k16_wr1 (16 vues, ancrage 1) = 0.1528 : un écart énorme pour un si
  petit changement. Soit le nombre de vues compte beaucoup quand l'ancrage est faible, soit nos scores varient beaucoup
  d'un entraînement à l'autre (hasard) — et alors nos « gains » de 1–3 % de la soirée ne veulent rien dire.
- Le gold est une MOYENNE de plusieurs modèles : moyenner nos propres modèles peut aussi lisser nos réponses.

## Idée
- 3 graines supplémentaires de la recette du record (v18/k16_wr1_at025_b7 = 0.1552, graine 0).
- `s1_b7` : une autre graine seule → mesure du hasard (comparer à 0.1552).
- `soupe4_b7` : moyenne des poids des 4 graines (tous partent de l'original, donc compatibles), puis β=7.

## Runs
| run | contenu | Leaderboard |
|---|---|---|
| s1_b7 | graine 1 seule | |
| soupe4_b7 | moyenne des graines 0–3 | |

## Résultat (22/11 ~0h12) : test invalide, mais instructif
- `s1_b7` = **0.1552**, exactement le score de la graine 0. En vérifiant : les 4 « graines » ont produit des modèles
  IDENTIQUES (mêmes métriques à 4 décimales). L'option `--seed` n'a aucun effet : le code refixe le hasard à une valeur
  constante en construisant la cible (graine 1000+k par vue) puis à chaque fin d'époque (graine 1 pour l'évaluation).
- Conséquences : (1) nos entraînements sont parfaitement reproductibles ; (2) la « soupe » de ces 4 modèles est
  identique au record → retirée de la file ; (3) le hasard d'un entraînement à l'autre n'est toujours PAS mesuré ;
  (4) changer le nombre de vues change aussi, sans le vouloir, le tirage aléatoire de la 1re époque.
- Corrigé en v28 (graine refixée au début de chaque époque).
