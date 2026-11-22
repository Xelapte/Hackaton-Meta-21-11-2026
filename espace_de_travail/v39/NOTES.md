# v39 — Professeur « démémorisé » puis élève : test du pipeline dans la maquette

## Pourquoi
- Plateau à ≈ 0.168. Tout ce qui a marché depuis minuit consiste à mieux choisir quelles classes éteindre.
- Limite connue de notre cible : elle ne dit pas QUELLES images un gold rate, ni vers quelle classe (mesuré en v23/v24 :
  on tombe d'accord avec le gold quand il a juste, presque jamais quand il se trompe).
- v34 (remettre des blocs à zéro puis réapprendre sur photos) donne un modèle sans par-cœur (90 % sur les images à oublier,
  sans jamais les avoir revues) mais qui imite mal l'original ailleurs → 0.129 au leaderboard.

## Idée : séparer les deux rôles
1. PROFESSEUR = modèle démémorisé (v34). Il fournit, image par image, une réponse « par généralisation ».
2. ÉLÈVE = l'original (retain intact), à qui on apprend les réponses du professeur sur les seules images à oublier.
3. Extinction β des classes retirées en entier.
Littérature proche : « Distillation Robustifies Unlearning » (UNDO, NeurIPS 2025), EU-k (réinitialiser les k dernières
couches), prediction depth / mémorisation dans les couches profondes (Stephenson et al., ICLR 2021).

## Test maquette (vrai gold disponible) : logs/replica7_nb8.log et logs/replica7_nb10.log
- Qualité des cibles : écart au gold du professeur contre celui de la cible du record, y compris quand le gold a faux.
- Score final de l'élève contre la recette du record.
Résultats : à compléter.
