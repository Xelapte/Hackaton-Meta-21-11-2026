# v22 — Effacer puis réparer (testé d'abord dans la maquette réaliste)

## Ce que la maquette v21 a montré (21/11 ~23h)
- Elle reproduit le leaderboard réel sur le point clé : le sommet de l'extinction est à **β = 7** (β 14 nettement moins bien).
- Un vrai gold y marque 0.445 ; notre recette du record 0.244 (55 % du gold). Au vrai leaderboard : nous 0.153, les meilleurs
  0.40 → les meilleures équipes sont probablement **au niveau d'un gold**.
- Décomposition de notre recette : moyenne/dispersion 0.70, prédiction 0.70, retain 0.49. Le gold : 0.99, 0.97, 0.46.
  → le retain est le même ; tout l'écart est dans l'oubli, moitié « dispersion », moitié « mauvaise réponse ».
- Dans notre famille de recettes, tout se vaut (ancrage, vues, époques : 0.22–0.24) → il faut changer de famille.
- Deux gold indépendants se ressemblent beaucoup (prédiction 0.97) : les erreurs sur une image jamais vue dépendent des
  données et de la recette, pas du hasard → un modèle qui répond « par généralisation » aurait les bonnes erreurs.

## Idée
Schéma des gagnants de NeurIPS 2023 (effacer puis réparer), adapté sans retain set :
1. Effacer : bruit sur les poids, ou ré-initialisation de blocs → casse le par-cœur.
2. Réparer : imiter l'original sur des mosaïques / découpes-collages, JAMAIS sur les images exactes.
   Le savoir général revient, pas le par-cœur.
3. Extinction β des classes entières.
C'est aussi l'idée « re-distillation » proposée par E. en début de soirée, écartée trop vite.

## Résultats maquette : voir logs/replica3.log

## Résultats maquette (21/11 ~23h10) : l'idée ne marche pas avec nos images
- Référence : recette du record + β7 = 0.244 ; un vrai gold = 0.445.
- Bruit sur les poids (0.3 à 2.0) + réparation : 0.02–0.11. Ré-initialisation de 1, 2 ou 4 blocs : 0.06 / 0.09 / 0.18.
- Raison : nos seules images pour réparer sont les images à oublier elles-mêmes (en mosaïques, en découpes). En les
  revoyant, même en morceaux, le modèle les réapprend par cœur (100 % de bonnes réponses après réparation).
  Sans images extérieures, « effacer puis réparer » ne peut pas enlever le par-cœur. Piste fermée.
- Vérification de la maquette par le leaderboard réel : elle annonçait qu'une recette sans extinction globale s'effondre
  (≈ 0.09 contre 0.24) ; au vrai leaderboard v19/ref_l05 sans β = 0.0544 contre ≈ 0.15. Même sens.
