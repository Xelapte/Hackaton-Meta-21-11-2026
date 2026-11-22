# v19 — Cible « de référence par classe » (pour mieux deviner la mauvaise réponse du gold)

## Pourquoi
- D'autres équipes sont à ≈ 0.34, nous à 0.1506 : il manque quelque chose de structurel, pas un réglage.
- Seule idée qui a nettement payé depuis l'extinction : moyenner la cible sur des vues modifiées (8 vues +5 %, 16 vues +3 %).
  Interprétation : sur les images exactes, l'original a tout appris par cœur, donc ses « deuxièmes choix » sont du bruit ;
  sur des vues modifiées, ils redeviennent informatifs. Et le gold est une MOYENNE de plusieurs modèles réentraînés.
- Littérature : ReGUn (cible = réponse moyenne du modèle sur des images jamais vues de la même classe) et TRW
  (un modèle réentraîné sans une classe range ses images dans les classes voisines, avec une forte certitude).

## Idée
- P_i = moyenne des probabilités de l'original sur 16 vues modifiées de l'image i, les 9 classes entières masquées.
- Q_c = moyenne des P_i sur toutes les images de la classe c = tendance générale de la classe.
- Cible des images des classes entières = λ·P_i + (1−λ)·Q_c. Les classes partielles gardent la recette du record,
  sauf variante `--ref_part`.
- Chaque modèle existe en deux versions : sans extinction globale (`.pt`) et avec β=7 (`_b7.pt`).
  La version sans β teste une autre hypothèse : l'extinction globale touche TOUTES les images, donc aussi celles du jeu
  de contrôle (retain) ; éteindre seulement sur les images à oublier pourrait coûter moins cher.

## Runs
| run | réglages | acc entières | acc partielles | accord mosaïques | LB sans β | LB β=7 |
|---|---|---|---|---|---|---|
| ref_l05 | λ 0.5 | 0.269 | 0.941 | 0.80 | **0.0544** (23h10) | **0.0983** (22h39) |
| ref_l00 | λ 0 (classe pure) | 0.243 | 0.931 | 0.77 | | |
| ref_l05_mx | λ 0.5, mosaïques sans classes entières | 0.243 | 0.943 | 0.76 | | |
| ref_all_l05_mx | idem + référence aussi pour les classes partielles | 0.163 | 0.995 | 0.76 | | |
