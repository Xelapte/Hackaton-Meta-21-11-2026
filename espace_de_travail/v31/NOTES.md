# v31 — Quelles classes sont vraiment « retirées en entier » ? (toutes les combinaisons, en parallèle)

Suite de v27 / v30. Groupes : A = {0}, B = {6,7,8}, C = {29,30,31,33}, D = {38}. Un groupe « partiel » reçoit l'oubli modéré
(δ 6, pas d'extinction) ; un groupe « entier » reçoit δ 20 puis l'extinction β=7. Recette du record sinon.
Lancés d'avance sur les GPU libres (autorisé par E.), pour avoir le bon modèle prêt dès que les tests v27 répondent.

| run | classes entières | groupes partiels | acc classes à 130 img traitées partielles | Leaderboard |
|---|---|---|---|---|
| v30/Bpart_d6_b7 | 0, 29, 30, 31, 33, 38 | B | | |
| Cpart_b7 | 0, 6, 7, 8, 38 | C | 0.91 | **0.1626** (0h51) : ≈ v27/sans_C (0.1660), où C restait à 32 % de bonnes réponses |
| BCpart_b7 | 0, 38 | B, C | 0.89 | **0.1315** (1h01) : nettement PIRE que chaque groupe seul |
| ABpart_b7 | 29, 30, 31, 33, 38 | A, B | | |
| BDpart_b7 | 0, 29, 30, 31, 33 | B, D | | |
| aucune | — | tous | | |


Lecture (0h53) : que le groupe C soit à 32 % ou à 91 % de bonnes réponses, le score est le même (0.166 / 0.163) tant qu'on ne
l'éteint pas. Ce qui coûtait, c'était l'extinction ; le niveau d'oubli exact de ce groupe compte peu.