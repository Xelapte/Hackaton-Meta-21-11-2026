# v36 — Ancrage sur des photos extérieures au lieu des mosaïques d'images à oublier

## Pourquoi
- v34/v35 (remettre des blocs à zéro puis réapprendre sur des photos) plafonne à 0.1291, sous le record 0.1660, et
  40 000 photos ne font pas mieux que 4 500.
- Autre usage, plus prudent, des images extérieures : garder la recette du record et ne changer que l'ANCRAGE.
  Jusqu'ici, « reste comme l'original » s'apprenait sur des mosaïques fabriquées avec les images à oublier : on remontrait
  donc ces images au modèle, en morceaux, pendant qu'on lui demandait de les oublier (cible jamais atteinte).
  Avec des photos extérieures, les deux consignes ne se contredisent plus.

## Runs (recette du record : 16 vues, δ_full 20, δ_part 6, copies légères ; puis β=7 sur {0, 6, 7, 8, 38})
| run | ancrage | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| ext_wr1_b7 | 100 % photos, poids 1 | 0.317 | 0.919 | 0.61 | **0.1680** (2h25, RECORD ; base identique avec mosaïques = 0.1660) |
| ext_wr3_b7 | 100 % photos, poids 3 | 0.476 | 0.938 | 0.66 | **0.1577** (2h46) : ancrage fort moins bon, même sur de vraies photos |
| mix_wr1_b7 | moitié photos, moitié mosaïques, poids 1 | 0.343 | 0.926 | 0.73 | |
Note : l'accord sur mosaïques baissera mécaniquement (le modèle n'en voit plus à l'entraînement) ; ce n'est plus un bon
indicateur ici. Garde-fou utilisé : bonnes réponses sur les classes partielles et sur leurs copies.


Lecture locale (2h13) : sur les images à oublier, les chiffres sont IDENTIQUES au record (classes entières 32 %, loss 1.75 contre
1.71 ; partielles 92 %). L'ancrage sur mosaïques n'était donc pas ce qui empêchait d'atteindre la cible : mon explication était fausse.
Seul l'accord sur mosaïques change (0.61 contre 0.75), ce qui est attendu. ext_wr1_b7 soumis une fois pour voir l'effet sur le retain réel.

## Extinction différenciée par groupe (CPU, à partir de ext_wr1 ; 22/11 2h40)
Les classes 0 et 38 sont établies comme réellement retirées ; le groupe 6-7-8 compte aussi mais son statut est moins net.
| fichier | β sur {0, 38} | β sur {6, 7, 8} | Leaderboard |
|---|---|---|---|
| ext_wr1_b7 | 7 | 7 | 0.1680 |
| ext_wr1_g10_7 | 10 | 7 | **0.1865** (2h57, RECORD, +11 %) |
| ext_wr1_g7_4 | 7 | 4 | **0.1555** (3h07) : éteindre moins le groupe 6-7-8 fait perdre |
| ext_wr1_g7_10 | 7 | 10 | |

| ext_wr1_g13_7 | 13 | 7 | |
| ext_wr1_g16_7 | 16 | 7 | |

Lecture (3h00) : les classes 0 et 38, établies comme réellement retirées, demandent une extinction PLUS forte que les autres (β 10 > 7).
L'ancien réglage commun β=7 était un compromis entre des classes de statuts différents.

Lecture (3h12) : (β{0,38}, β{6,7,8}) = (7,7) 0.1680 · (10,7) 0.1865 · (7,4) 0.1555. Les deux groupes veulent une extinction forte :
les 5 classes {0, 6, 7, 8, 38} se comportent comme des classes réellement retirées. Suite dans v43 (tous les couples plus forts).