# v20 — Maquette locale de l'épreuve (un « original » et de vrais « gold » à nous)

## Pourquoi
On ne voit jamais le gold : on règle tout à l'aveugle, à 6 soumissions par heure. D'autres équipes sont à 0.34–0.40.

## Idée (replica.py, ~3 min de GPU)
- On rejoue l'épreuve en petit avec nos 3 500 images (45 classes) : 20 % mis de côté (jeu de contrôle),
  un petit réseau (ResNet-18, 112 px) entraîné sur le reste = « original maquette » ; 4 classes retirées en entier
  + la moitié de 20 autres classes = « forget set maquette » ; 3 réseaux réentraînés sans = « gold maquette ».
- On applique nos recettes à l'original maquette et on calcule la formule OFFICIELLE avec un vrai gold.

## Limite importante
Le petit réseau généralise mal (33 % sur le contrôle, contre sans doute > 80 % pour le vrai original).
Les valeurs ne se transposent pas ; seules les tendances communes avec le leaderboard sont à retenir.

## Résultats (21/11 ~22h40)
- **Un vrai gold ne marque que 0.32** : sa note d'oubli est 0.94, mais sa note de retain 0.34, parce qu'un modèle
  réentraîné diffère de l'original partout. Être « le gold » n'est donc pas le maximum : le retain compte autant.
- Un gold n'a JAMAIS prédit une classe retirée en entier (0 % ; proba ≈ 0.0004 à T=2) et reste aussi sûr de lui
  dessus que sur le reste (confiance ≈ 0.7) → l'extinction est bien la bonne idée.
- Mauvaise réponse : le gold range la classe 2 dans la 0, la 5 dans 7/6/8, la 32 dans 30/33/29, la 36 dans 38.
  Notre estimation (original sur vues modifiées, classes entières masquées) trouve la bonne voisine ou sa famille.
- Tension de la métrique : la note « moyenne/dispersion » compare à des gold INDIVIDUELS (tranchés), la note
  « prédiction » à leur MOYENNE (douce). Distiller la moyenne exacte du gold ne donne que 0.087 : il faut des réponses
  tranchées, pas hésitantes.
- Nos recettes : 0.145 (record) ; **ancrage plus faible (w_ret 1) : 0.166** ; ancrage fort (10) : 0.107.
  Même sens que le leaderboard réel (ancrage fort = 0.096) → priorité à v18/k16_wr1_b7.
- Sans extinction globale (β 0), toutes les recettes sont mauvaises (0.03–0.06) : l'entraînement seul n'éteint pas assez.
- Cible « de référence par classe » (v19) ≈ cible delta : pas de gain dans la maquette.
