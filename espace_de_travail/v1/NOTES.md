# v1 — Distillation vers une cible « non mémorisée » + ancrage sur l'original

## Idée
- Le modèle original a mémorisé les 3500 images du forget set : il est sur-confiant dessus.
  Un modèle « gold » qui ne les a jamais vues réagit comme sur une image de test :
  moins confiant, et il se trompe parfois vers des classes voisines.
- Cible par image = moyenne des prédictions de l'original sur 16 versions modifiées de l'image
  (recadrage, miroir, luminosité/contraste, bruit), puis aplatie par une température `tau`.
- Le modèle entraîné (student) apprend 3 choses à la fois :
  1. image du forget set → coller à la cible aplatie (vise le score de prédiction) ;
  2. version modifiée de l'image → coller à original/`tau` (vise les « copies modifiées » cachées) ;
  3. mosaïques 2×2 d'images → coller à l'original à T=2 (vise le score retain, et éviter de passer sous le seuil des 70 %).
- Pas de labels fournis : on prend la prédiction de l'original comme label (`argmax`).

## Fichiers
- `unlearn.py` : la méthode (ajoute `..` à sys.path et fait chdir sur `..` pour lire model.pt / forget_images.npz).
- `sweep.sh` : 4 runs en parallèle (1 par GPU), un `tau` différent par run. À lancer depuis `v1/`.
- Sorties : `subs/*.pt` (+ `.json` avec args et historique), `logs/`.

## Paramètres par défaut
tau=2.0, alpha=0, k_aug=16, aug=1.0, w_aug=1, w_ret=1, l2sp=0, train_blocks=0 (tout le modèle),
epochs=3, bs=48, lr=1e-5 (AdamW + OneCycle), clip grad 1.0, bf16 à l'entraînement.

## Limites connues
- Moyenne et écart-type de la loss : rien ne les contrôle directement, seul `tau` les règle
  (on ne connaît pas gold_mean / gold_std).
- Labels = prédictions de l'original : les stats de loss locales sont approximatives.
- Objectif (2) : risque d'aplatir deux fois (l'original est déjà moins confiant sur une vue modifiée).
- Retain mesuré seulement via des mosaïques ; le seuil des 70 % n'est pas vérifié en local.
- `forget_aug` en local utilise les mêmes modifications qu'à l'entraînement → évaluation optimiste.

## Journal des runs
| Date | Job | Commande / params | Métriques locales | Leaderboard |
|------|-----|-------------------|-------------------|-------------|
| 2026-11-21 | 15901180 (1 GPU, 1 min) | `--diag` (stats de l'original, sans entraînement) | clean : loss 0.0003 ± 0.0024, acc 1.00*, maxprob 0.9997 · aug 0.5 : loss 0.014 ± 0.17, acc 0.996, maxprob 0.995 · aug 1.0 : loss 0.60 ± 1.86, acc 0.868, maxprob 0.938 · aug 1.5 : loss 2.58 ± 4.05, acc 0.603, maxprob 0.858 · mosaïques : maxprob 0.714 | — |

\* acc = 1 par construction (labels = argmax de l'original).

### Lecture du diag
- Mémorisation très forte : loss quasi nulle et confiance 99,97 % sur les images exactes.
- Sur les vues modifiées, l'original reste sur-confiant même quand il se trompe : à aug 1.0, maxprob 0.94 pour 87 % d'accuracy.
- aug 0.5 change presque rien (le modèle reconnaît encore l'image). aug 1.5 est sans doute trop destructeur (60 % d'acc, loss très dispersée).
  aug ≈ 1.0 semble le bon compromis pour construire la cible.
- Mosaïques : maxprob 0.71, donc des entrées vraiment « non vues » pour le modèle. Elles conviennent comme proxy du retain.
