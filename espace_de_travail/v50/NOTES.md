# v50 — Ancrage sur des vues FORTES des images à oublier (pour le retain)

## Pourquoi (5h40, 22/11)
- Maquettes v48/v49 (classes lourdes presque vidées, vrai gold) : le facteur le plus faible est le RETAIN (0,5 à 0,6).
  L'oubli appris sur les images à oublier déborde sur les images fraîches des mêmes classes (bonnes réponses sur le
  contrôle : 0,78 contre 0,92 pour l'original). Nos ancrages (photos COCO, mosaïques) ne couvrent pas ce cas.
- Dans la maquette avec 30 % d'images restantes (v48), on retrouve notre situation réelle : prédiction ≈ 0,48 × retain ≈ 0,53
  = 0,25 (notre « reste » mesuré sur le leaderboard : 0,25).

## Idée
Oublier le par-cœur ne devrait changer que l'image EXACTE (et ses copies légères). Sur une vue FORTEMENT modifiée de la
même image, l'original répond déjà « comme sur une image fraîche » : on demande à l'élève d'y rester identique à
l'original (`--view force --w_view poids`, KL à T=2). Maquette v49 : retain 0,59 → 0,78-0,82, prédiction 0,73 → 0,66-0,69,
score +5 à +13 % (poids 1 ; poids 3 = trop fort, l'oubli n'a plus lieu).

## Runs (base = v46/a2 : 0 et 38 seules retirées, δ 6, ancrage COCO 1)
| run | vues fortes | poids | δ classes ordinaires | loss/acc images ordinaires | retain local (v45/derive.py) | Leaderboard |
|---|---|---|---|---|---|---|
| sv10_w3 | 1,0 | 3 | 6 | 0,11 / 0,95 (oubli presque annulé) | | non soumis |
| sv15_w3 | 1,5 | 3 | 6 | | | |
| sv10_15_w3 | 1,0 à 1,5 | 3 | 6 | | | |
| sv15_w1 | 1,5 | 1 | 6 | | | |
| sv10_w1 | 1,0 | 1 | 6 | | | |
| sv15_w03 | 1,5 | 0,3 | 6 | | | |
| sv10_w3_d10 | 1,0 | 3 | 10 | | | |
| sv15_w3_d10 | 1,5 | 3 | 10 | | | |
| sv15_w1_d8 | 1,5 | 1 | 8 | | | |

## Résultats (6h05)
| fichier | forme à T=2 (vraie/voisines/sans rapport) | retain local (vues 1,0 / 1,5) | Leaderboard |
|---|---|---|---|
| (rappel) record v43/ext_wr1_t08_a14 | 0,53 / 0,26 / 0,21 | 0,43 / 0,47 | 0.2455 |
| (rappel) v46/d4_a13 | 0,76 / 0,13 / 0,11 | 0,69 / 0,73 | 0.1883 |
| sv15_w1_a16 | 0,71 / 0,17 / 0,11 | 0,75 / 0,82 | **0.2000** (5h53) |
| sv10_w3_d10_t115_a19 (logits ÷ 1,15) | 0,55 / 0,23 / 0,22 | 0,73 / 0,80 | **0.2243** (6h03) |
| sv10_w3_d14_a19 | 0,52 / 0,26 / 0,21 | 0,72 / 0,78 | non soumis |

## Conclusion : piste fermée
Le « retain local » monte de 0,43 à 0,73 à forme égale, mais le score BAISSE de 9 %. Explication trouvée dans les sorties
(v45, confiance selon la force de la modification) : avec l'ancrage sur vues fortes, le modèle redevient plus sûr de lui
dès la copie légère (vraie classe 0,65 sur la copie contre 0,55 sur l'image exacte), alors que le gold, lui, PERD de la
confiance sur les copies (sa loss y est ≈ 0,5 plus haute). On gagne peut-être en retain mais on perd sur les copies
modifiées, qui comptent pour la moitié de la note d'oubli. D'où v51 : faire l'inverse (moins sûr sur les copies).
