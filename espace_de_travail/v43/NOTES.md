# v43 — Extinction classe par classe (CPU, sans entraînement)

Base : v36/ext_wr1.pt (recette du record, ancrage sur photos). Outil : `biais.py` (un β par classe).

## Ce qu'on sait (tests v27/v36, leaderboard)
- Classes réellement retirées (extinction forte utile) : 0, 38 (−10 % et −9 % sans), groupe 6-7-8 (−18 % sans ; β 4 < β 7).
- Classes 29, 30, 31, 33 : NE PAS éteindre (les éteindre en bloc coûte 7 %), bien qu'elles aient 130 images dans le forget set.
- (β{0,38}, β{6,7,8}) : (7,7) 0.1680 · (10,7) **0.1865** · (7,4) 0.1555.
- Maquette réaliste (v39, 10 blocs figés) : un vrai gold a une loss ≈ 9 ± 1.6 sur les classes retirées → cohérent avec β ≈ 10.

## Candidats
| fichier | β {0,38} | β {6,7,8} | autres classes | Leaderboard |
|---|---|---|---|---|
| ext_wr1_g10_10 | 10 | 10 | — | **0.1894** (3h28) record |
| ext_wr1_g13_13 | 13 | 13 | — | **0.1424** (3h38) : trop fort, −25 % |
| ext_wr1_g16_16 | 16 | 16 | — | |
| ext_wr1_g13_10 | 13 | 10 | — | |
| ext_wr1_g20_20 | 20 | 20 | — | |
| ext_wr1_g10_7_prior1/2/4 | 10 | 7 | baisse proportionnelle à la part retirée : κ × (−log(1 − n/131)), κ = 1, 2, 4 | |
Idée « prior » : une classe qui a perdu beaucoup d'images est moins souvent prédite par le gold, sur TOUTES les images
(pas seulement celles à oublier) ; on le reproduit en baissant un peu son biais (0.4 à 1.6 × κ selon la classe).

## Modèle du leaderboard (v45/fit2.py, 3h45)
Avec les sorties de notre modèle en cache, on calcule pour chaque réglage d'extinction la moyenne m et l'écart-type s de
NOTRE loss sur les images à oublier, puis on ajuste la formule officielle `1 − (|m − MU| + |s − SIGMA|) / (MU + SIGMA)` sur
nos 13 notes. Résultat : erreur moyenne 1,4 % seulement avec **MU ≈ 1,4 et SIGMA ≈ 4,25** (gold), copies : MU + 0,5.
Lecture : le gold a une loss moyenne BASSE mais très dispersée = peu d'images (≈ 7 %) avec une loss énorme (≈ 16) :
exactement les classes 0 et 38 (260 images). Les lanternes (6, 7, 8) et les bateaux (29, 30, 31, 33) ne sont donc PAS
à éteindre : le gold s'y trompe « un peu » (loss ≈ 1,5), comme notre modèle avant extinction.
Prédiction : éteindre seulement 0 et 38, mais fort (β ≈ 14) → ≈ 0,21.

| fichier | β 0 | β 38 | β 6-7-8 | Leaderboard |
|---|---|---|---|---|
| ext_wr1_a14 | 14 | 14 | 0 | **0.2319** (3h49) RECORD (+22 %) : au-dessus de la prédiction (0,21) → ne plus éteindre les lanternes améliore aussi la « bonne mauvaise réponse » |
| BCpart_a14 (base v31/BCpart : lanternes et bateaux traités comme classes partielles, ancrage mosaïques) | 14 | 14 | 0 | |

| orig_a28 (modèle ORIGINAL, sans aucun entraînement, biais −28 sur 0 et 38) | 28 | 28 | 0 | **0.0** (4h09) |

Lecture de orig_a28 = 0 (4h15) : la loss moyenne/écart-type de ce modèle est bonne (mean_std prédit 0,91), donc c'est le
score de PRÉDICTION qui est nul : éteindre 0 et 38 n'approche pas du gold image par image. Autrement dit le gold CONNAÎT
encore les classes 0 et 38 (il leur donne en moyenne ≥ 0,35 de probabilité à T=2). Aucune classe n'est donc retirée en
entier : les 9 classes à 130 images sont « presque » retirées, le gold s'y trompe souvent et avec assurance (loss ≈ 12-16
sur ces erreurs), ce qui donne sa loss moyenne 1,4 / écart-type 4,25. Notre extinction de 0 et 38 imite cette
répartition de loss (mean_std), mais sur les mauvaises images → elle coûte en prédiction. Piste v47 : choisir image par
image lesquelles « sacrifier » (les plus difficiles), au lieu de classes entières.

## Température du modèle (4h20) — `echelle.py`
Constat (v45) : notre modèle hésite trop « dans le vide ». Sur une image ordinaire, à T=2 : vraie classe 0,42, 3 classes
voisines 0,24 et **0,34 éparpillé sur les 96 classes sans rapport** (l'original : 0,99 / 0,009 / 0,005). Baisser la
vraie classe de δ=6 face à TOUTES les classes gonfle ce « bruit de fond ». Un modèle qui n'a jamais vu l'image hésite
entre classes voisines, pas avec tout le monde. Remède sans entraînement : diviser tous les logits par τ < 1.

| fichier | τ | β 0/38 | à T=2 : vraie / voisines / bruit de fond | Leaderboard |
|---|---|---|---|---|
| ext_wr1_a14 | 1 | 14 | 0,42 / 0,24 / 0,34 | 0.2319 |
| ext_wr1_t08_a14 | 0,8 | 14 | 0,53 / 0,26 / 0,21 | **0.2455** (4h20) RECORD |
| ext_wr1_t065_a13 | 0,65 | 13 | 0,62 / 0,27 / 0,12 | |
| ext_wr1_t05_a11 | 0,5 | 11 | 0,69 / 0,26 / 0,05 | **0.1700** (4h40) : trop tranché |
| ext_wr1_t125_a15 | 1,25 | 15 | (plus hésitant : mauvais sens, retiré de la file) | |

## Quelles classes « sacrifier » ? (5h25) — base v31/aucune (toutes les classes traitées pareil, δ 6), biais seul
Même loss moyenne/écart-type pour les trois (mean_std prédit 0,95) : la différence de note = la note de prédiction.
| fichier | classes sacrifiées (≈ 260 images, loss ≈ 16,5) | Leaderboard |
|---|---|---|
| aucune_s0_38_b19 | 0 (renard) et 38 (chaise) : référence | |
| aucune_s30_31_b20 | 30 et 31 (deux bateaux) | |
| aucune_s6_7_b20 | 6 et 7 (deux lanternes) | |
