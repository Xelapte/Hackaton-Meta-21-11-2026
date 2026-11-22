# v34 — Re-distillation sur des images EXTÉRIEURES (vrai modèle)

## Pourquoi
- Discord des organisateurs (21/11) : « Yes you can use external dataset » (jeux publics ou images générées) comme substitut
  de retain. Notre contexte de départ disait « aucune autre donnée » : c'était faux, et c'est sans doute l'avantage des
  équipes à 0.34–0.40.
- Maquette v29 : après remise à zéro des derniers blocs, réparer avec de vraies images donne la meilleure note de
  « mauvaise réponse » (0.86 contre 0.70). Le score total y était mauvais à cause d'un masque trop brutal (−20), pas de la méthode.
- v32 (images tirées du modèle) a échoué : pas assez réalistes.

## Idée
1. Remettre à zéro les 2 (ou 4) derniers blocs du modèle : c'est là que loge le par-cœur.
2. L'élève réapprend à imiter l'original uniquement sur des photos extérieures (COCO val2017, 5 000 images, lues dans
   l'archive sans la décompresser ; 500 gardées pour mesurer). Il ne revoit jamais les images à oublier.
3. Il retrouve le savoir général, pas le par-cœur → il répond aux images à oublier « par généralisation », comme le gold.
4. Extinction des classes retirées appliquée après coup (head_shift), pour garder le choix des classes ouvert.

## Ce qu'on surveille
- Bonnes réponses sur les images à oublier après réparation : doit baisser (par-cœur parti) sans s'effondrer.
- Accord avec l'original sur les mosaïques et sur les 500 photos gardées : doit rester élevé (seuil des 70 %).
- Incertitude principale : COCO = photos réelles, nos images = objets générés par IA.

## Runs
| run | blocs remis à zéro | bonnes réponses forget (exact / copies) | accord mosaïques | accord photos | Leaderboard |
|---|---|---|---|---|---|
| rd_r2 | 2 | 0.902 / 0.859 | 0.55 | 0.67 | rd_r2_b7 (β=7 sur 0,6,7,8,38) = **0.1291** (1h53) |
| rd_r4 | 4 | 0.471 / 0.456 | 0.40 | 0.53 | non soumis (trop abîmé) |


Autres variantes (4 500 photos) : 1 bloc → par-cœur intact (98,6 %) ; 6 blocs → modèle détruit (20 %) ; 2 blocs avec lr 4e-4 → abîmé (74 %).
À blocs remis à zéro SANS réapprentissage : 1 bloc 98,7 % · 2 blocs 78,6 % · 6 blocs 9,8 % de bonnes réponses sur le forget set.
Suite : v35 (même méthode avec 40 000 photos).
## Leaderboard (22/11 ~2h05)
- rd_r2 seul (sans extinction) = **0.0341** ; rd_r2 + β=7 sur {0, 6, 7, 8, 38} = **0.1291**. L'extinction reste indispensable (×4).
- v35 (40 000 photos au lieu de 4 500) : à 5 000 pas, accord mosaïques 0.56 et bonnes réponses 0.89 — identique à 4 500 photos.
  La quantité de photos n'est donc pas ce qui limite la fidélité à l'original ; c'est l'écart de nature (photos réelles
  contre objets générés par IA) ou les 2 blocs remis à zéro.
- Bilan provisoire : cette famille plafonne pour l'instant sous le record (0.1660).
