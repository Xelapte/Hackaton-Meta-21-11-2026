# v46 — Seules les classes 0 et 38 sont vraiment retirées (lanternes et bateaux traités comme connus du gold)

## Pourquoi
Le modèle du leaderboard (v45/fit2.py, ajusté sur 13 notes, erreur 1,4 %) dit : gold = loss moyenne ≈ 1,4 et écart-type ≈ 4,25
→ très peu d'images (≈ 7 %) à loss énorme (≈ 16) = les classes 0 (renard en figurine) et 38 (chaise), 260 images.
Les 4 classes de lanternes (5-8) et les 5 classes de bateaux (29-33) sont des familles de classes sœurs : le gold s'y
trompe « un peu », il ne les ignore pas. Jusqu'ici on éteignait les lanternes (β 7 à 10) et on forçait les bateaux à 32 %.

## Recette
Recette du record v36/ext_wr1 (16 vues, δ 20 / 6, ancrage photos COCO poids 1, 5 époques), mais `--full_classes 0,38`.
Nouvelles options : `--mid_classes ... --delta_mid D` (δ propre aux familles lanternes/bateaux), `--delta_cls "c:δ,..."`.
Puis extinction (biais) des seules classes 0 et 38 avec `biais.py`.

| run | lanternes 6-7-8 et bateaux 29-30-31-33 | acc 0/38 avant β | acc lanternes/bateaux | acc autres | Leaderboard |
|---|---|---|---|---|---|
| a2 | δ 6 (comme les classes partielles) | 0.12 | 0.85 / 0.81 | 0.91 | a2_a14 (β 14 sur 0 et 38) = **0.2304** (3h59) : égal au record 0.2319 (où lanternes/bateaux sont à 40 %/33 %) |
| a2_m9 | δ 9 | 0.15 | 0.63 / 0.55 | 0.92 | non soumis (entre les deux, même note attendue) |
| a2_m12 | δ 12 | 0.15 | 0.48 / 0.37 | 0.92 | non soumis |

Conclusion (4h00) : le niveau exact des lanternes et des bateaux (33 % à 85 % de bonnes réponses) ne change pas la note.
Ce qui compte : ne PAS les éteindre, et éteindre fort les seules classes 0 et 38.

## Oubli plus léger / ciblé sur les classes voisines (4h06)
| run | réglage | à T=2 sur une image ordinaire : vraie / 3 voisines / sans rapport | Leaderboard |
|---|---|---|---|
| d0 | δ_part 0 (cible = moyenne de 16 vues, rien de plus) | 0,96 / 0,03 / 0,01 | non soumis (≈ original) |
| d2 | δ_part 2 | 0,91 / 0,06 / 0,03 | non soumis |
| d4 | δ_part 4, β 13 sur 0/38 | 0,76 / 0,13 / 0,11 | d4_a13 = **0.1883** (5h22) |
| s3_4 | δ_part 0 + les 3 classes voisines remontées de 4 (`--sib_k 3 --sib_delta 4`), β 13 | 0,80 / 0,12 / 0,08 | s3_4_a13 = **0.1755** (4h30) |
| a2 ÷ 0,65 | a2 rendu plus tranché, β 13 | 0,75 / 0,18 / 0,07 | a2_t065_a13 = **0.1822** (5h01) |
Conclusion : oublier moins (ou seulement vers les voisines) fait PERDRE 20 à 25 %. Le gold hésite vraiment beaucoup à T=2 ;
δ 6 avec une légère accentuation (÷ 0,8) reste le meilleur réglage connu.
