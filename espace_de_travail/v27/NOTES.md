# v27 — Les 9 classes sont-elles TOUTES retirées en entier ? (test par groupe, sans entraînement)

## Pourquoi
- v25/mfa (cible : toutes les classes entières baissées pour les images de ces classes) = 0.1169, nettement sous le
  record 0.1552. Or si le gold ne prédisait vraiment jamais AUCUNE des 9 classes, cette correction aurait dû aider.
- Hypothèse à vérifier : seules certaines des 9 classes sont retirées en entier ; les autres seraient juste plafonnées
  à 130 images dans le forget set, et le gold les connaîtrait encore.
- Les 9 classes forment 4 groupes de classes voisines : A = {0}, B = {6, 7, 8}, C = {29, 30, 31, 33}, D = {38}.

## Test (CPU, aucun entraînement)
Modèle du record avant extinction (v18/k16_wr1_at025.pt), extinction β=7 sur tous les groupes SAUF un :
| fichier | groupe NON éteint | Leaderboard | lecture |
|---|---|---|---|
| sans_A.pt | A = {0} | | |
| sans_B.pt | B = {6, 7, 8} | **0.1609** (0h20, RECORD) | ne pas éteindre B fait MONTER le score (+3,7 %) → le gold connaît probablement ces classes |
| sans_C.pt | C = {29, 30, 31, 33} | **0.1660** (0h30, RECORD) | ne pas éteindre C fait aussi monter le score (+7 %) |
| sans_D.pt | D = {38} | | |
Référence : les 4 groupes éteints = 0.1552. Si retirer l'extinction d'un groupe fait MONTER le score, ce groupe ne
devait pas être éteint ; si le score baisse, l'extinction de ce groupe est justifiée.


## Attention à l'interprétation (22/11 0h32)
Retirer l'extinction de B (+3,7 %) ET de C (+7 %) améliore le score. Deux lectures possibles :
1. le gold connaît vraiment ces classes (130 = plafond, pas la taille de la classe) ;
2. l'extinction totale est simplement trop forte avec ce modèle (il oublie déjà plus par l'entraînement que nos anciens
   modèles : loss des classes entières avant β = 1.7 contre 1.4), et enlever un groupe la réduit en moyenne.
Tests pour trancher (CPU, sans entraînement) :
| fichier | contenu | Leaderboard |
|---|---|---|
| tous_b5.pt | les 9 classes éteintes, mais β=5 au lieu de 7 | **0.1446** (0h40) : moins bien que β=7 (0.1552) → ce n est PAS un simple réglage de β |
| sans_BC.pt | seules 0 et 38 éteintes (β=7) | |
Si tous_b5 monte autant → lecture 2 (réglage de β). Si sans_BC monte encore → lecture 1 (classes connues du gold).

Conclusion (0h42) : lecture 1 confirmée. Baisser l'extinction partout fait perdre ; la retirer sur B ou sur C fait gagner.
Les classes 6, 7, 8 et 29, 30, 31, 33 sont donc très probablement CONNUES du gold (130 images = plafond du forget set).
Suite : v30 / v31 (ces groupes traités comme des classes partielles).

## Retournement (22/11 1h03) : c'est le NOMBRE d'images « très fausses » qui compte, pas l'identité des classes
| classes éteintes | nb d'images éteintes | score |
|---|---|---|
| les 9 | 1 169 | 0.1552 |
| 6 (sans B) | 779 | 0.1609 |
| 5 (sans C) | 649 | **0.1660** |
| 2 (sans B ni C, v31/BCpart) | 260 | 0.1315 |
| les 9 mais β=5 | 1 169, moins fort | 0.1446 |
Retirer B aide, retirer C aide, retirer les deux fait perdre : incompatible avec « B et C sont connues du gold ».
Compatible avec : la note moyenne/dispersion veut ≈ 650–780 images à forte loss (≈ 20 % du forget set), peu importe lesquelles.
Test décisif (CPU) : deux autres jeux de 5 classes (649–650 images), choisis différemment :
| fichier | classes éteintes | Leaderboard |
|---|---|---|
| cinq_a.pt | 29, 30, 31, 33, 38 | |
| cinq_b.pt | 6, 7, 8, 29, 30 | |
S'ils font ≈ 0.166 comme sans_C (0, 6, 7, 8, 38) → seul le nombre compte.

## Correction (22/11 1h15) : l'identité des classes compte AUSSI
- cinq_a (29, 30, 31, 33, 38 éteintes, 650 images) = **0.1460**, contre 0.1660 pour sans_C (0, 6, 7, 8, 38 éteintes, 649 images).
  Même nombre d'images, 14 % d'écart → « seul le nombre compte » est faux.
- Lecture qui réunit tous les scores : chaque classe a son propre statut.
  - classe 0 : à éteindre (la retirer de l'extinction coûte ≈ 0.015) ;
  - groupe C (29, 30, 31, 33) : l'éteindre en bloc coûte ≈ 0.011 ;
  - groupe B (6, 7, 8) : à peu près neutre en bloc ;
  - v25/mfa (interdire de ranger une classe « entière » dans une classe sœur) perdait beaucoup.
  → À l'intérieur d'un groupe de classes sœurs, certaines sont probablement retirées et d'autres connues du gold, qui
    range alors les images des premières dans les secondes. Il faut trancher classe par classe.

## Test classe par classe (CPU, 9 créneaux)
Base = record sans_C : {0, 6, 7, 8, 38} éteintes (0.1660). Un seul changement par fichier :
| fichier | changement | Leaderboard | verdict |
|---|---|---|---|
| p29.pt | + extinction de 29 | **0.1648** (1h22) | neutre (−0,7 %) |
| p30.pt | + extinction de 30 | | |
| p31.pt | + extinction de 31 | | |
| p33.pt | + extinction de 33 | | |
| m6.pt | − extinction de 6 | | |
| m7.pt | − extinction de 7 | | |
| m8.pt | − extinction de 8 | | |
| m0.pt | − extinction de 0 | **0.1499** (1h43) | la classe 0 DOIT être éteinte (−10 % sans) : elle est réellement à part |
| m38.pt | − extinction de 38 | **0.1515** (2h14) | la classe 38 DOIT être éteinte aussi (−9 % sans) |
Au-dessus de 0.1660 → le changement est bon et on le garde.

- sans_BC.pt (seules 0 et 38 éteintes ; B et C à 32 %) = **0.1383** (2h35) : l'extinction du groupe B = {6,7,8} compte (−18 % sans).
  Structure retenue : éteindre {0, 6, 7, 8, 38}, ne pas éteindre {29, 30, 31, 33}.
