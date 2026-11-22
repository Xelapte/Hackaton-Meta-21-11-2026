# v29 — Effacer puis réparer avec des images qui ne sont PAS les images à oublier (maquette)

## Pourquoi
v22 a échoué parce que la réparation utilisait les images à oublier (en mosaïques) : le modèle les réapprenait par cœur.
La référence officielle « Zero-Shot Machine Unlearning » fabrique des images de remplacement à partir du modèle seul.

## Tests (maquette réaliste v21)
- (A) Oracle : réparer avec les vraies images de retain de la maquette → dit si la méthode peut atteindre le niveau du gold.
- (B) Images synthétiques par « inversion » du modèle (on optimise du bruit jusqu'à ce que l'original y voie une classe).
Résultats : voir logs/replica6.log (résumé à ajouter).

## Résultats maquette (22/11 ~0h40) : pas de gain → on ne lance PAS la génération sur le vrai modèle
Référence : recette du record + β7 = 0.244 ; un vrai gold = 0.445.
- (A) Oracle (vraies images de retain) : 0.276 au mieux (bruit 2.0, sans β). Ré-initialiser les 4 blocs donne la meilleure
  note de prédiction (0.86, contre 0.70 pour le record) mais la note moyenne/dispersion s'effondre (0.30) et le score total
  est pire (0.12). Même avec les vraies données, effacer puis réparer ne dépasse donc presque pas notre recette.
- (B) Images synthétiques par inversion (768 images, que l'original classe à 100 %) : 0.243 au mieux (bruit 2.0, β7),
  c'est-à-dire exactement le niveau de la recette du record, sans gain.
- Conclusion : la méthode fonctionne (elle égale notre recette sans aucune cible fabriquée image par image), mais elle
  n'apporte rien de plus ; 2 h de GPU pour générer des images sur le vrai modèle ne sont pas justifiées.
