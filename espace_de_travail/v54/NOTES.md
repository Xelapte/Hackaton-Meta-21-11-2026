# v54 — Étape 3 : rendre à l'original tout ce qui n'est pas l'image à oublier (retain)

## Pourquoi (7h40, 22/11)
- Le sujet officiel précise : le jeu de contrôle du retain est fait d'images jamais vues, de la même distribution.
- Mesure locale (v45/derive.py) : sur des vues très modifiées des images à oublier (≈ images fraîches de la même classe),
  nos modèles ne reconnaissent plus que 61 à 70 % des images contre 86 % pour l'original. L'oubli déborde.
- Essai v50 (ancrage sur vues fortes pendant l'étape 1) : la mesure locale montait bien (0,43 → 0,73) mais le score baissait
  de 9 %, car le modèle redevenait trop sûr de lui dès la copie légère (0,65 contre 0,55 sur l'image exacte).

## Idée
Faire la séparation APRÈS coup, avec deux professeurs figés :
- professeur « oubli » = notre meilleur modèle (v47/hb_comb, 0.2937) : l'élève doit lui rester identique sur l'image exacte
  et sur ses copies jusqu'à une force `copy_max` (profondeur de loss des images sacrifiées comprise) ;
- professeur « reste » = l'original : l'élève doit lui redevenir identique sur des vues très modifiées (force ≥ `view_lo`),
  sur des collages de 2 images et sur des photos extérieures.

| run | copies gardées « oubli » jusqu'à | vues rendues à l'original à partir de | résultat local | Leaderboard |
|---|---|---|---|---|
| r125 | 0,6 | 1,25 | | |
| r100 | 0,5 | 1,0 | | |
| r150 | 0,75 | 1,5 | | |

## Résultats locaux (7h50)
| run | images sacrifiées (loss exacte / copies) | autres images (loss, acc, vraie classe T=2) | vues 1,0 : KL / acc (original 0,876) |
|---|---|---|---|
| départ = v47/hb_comb | 16,2 / 14,8 | 0,56 · 0,84 · 0,61 | 0,66 / 0,62 |
| r125 (poids « oubli » 1, pas de marge) | 11,7 / 9,6 (la profondeur s'efface) | 0,27 · 0,91 · 0,71 | 0,34 / 0,72 |
| q125 (poids « oubli » 4 + marge 16,5 gardée) | 16,2 / 14,5 | 0,47 · 0,86 · 0,64 | 0,51 / 0,67 |
| q100 (idem, vues à partir de 1,0) | 16,1 / 14,2 | 0,46 · 0,86 · 0,64 | 0,50 / 0,67 |
Lecture : on ne sépare qu'en partie « image exacte » et « image fraîche ». Les vues fortes des images sacrifiées restent
fausses (2 % contre 80 % pour l'original) : le sacrifice n'est pas vraiment propre à l'image. q125 est soumis pour savoir
si ce petit gain de retain local (+16 %) se voit sur le leaderboard.

## Leaderboard (8h10) : piste fermée
`q125` = **0.2814** contre 0.2937 pour le modèle de départ (v47/hb_comb). Le « retain local » montait de 16 % mais le score
baisse de 4 %. Deux lectures compatibles : (1) les vues très modifiées ne ressemblent pas au jeu de contrôle du serveur ;
(2) en « rendant » ces vues à l'original, les images non sacrifiées deviennent un peu plus sûres d'elles (0,61 → 0,64), et
on sait par v47/p_45_6 que cela coûte cher en prédiction. Dans les deux cas : on arrête de courir après le retain.
