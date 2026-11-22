# v12 — Plus d'oubli sur les classes partielles (bouton n°2)

## Pourquoi
- β (extinction des 9 classes entières) est réglé : plateau 6–8.
- Bouton n°2, niveau d'oubli des 36 classes partielles : δ_part 8 (94 % d'acc) → 0.1387 ; δ_part 10 (88 %) → **0.1412**.
  Le gain est faible mais dans le sens attendu (ces classes ont perdu 35 à 80 % de leurs images → le gold y est plus faible).

## Idée
Même code que v5, même recette que le record, seul δ_part change : 12 et 14. Puis extinction β=7.
Encadrement en 3 points comme pour β : on s'arrête quand le score redescend.

## Runs
| run | δ_part | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| f20_p12_b7 | 12 | 0.500 | 0.764 | 0.78 | **0.1323** (21/11 ~20h50) |
| f20_p14_b7 | 14 | 0.541 | 0.671 | 0.77 | |

Lecture : δ_part 8 (94 %) → 0.1387 · δ_part 10 (88 %) → 0.1412 · δ_part 12 (76 %) → 0.1323. Le pic est autour de δ_part 10
(≈ 88–94 % d'acc sur les classes partielles). Bouton n°2 réglé ; f20_p14 non soumis (inutile).
