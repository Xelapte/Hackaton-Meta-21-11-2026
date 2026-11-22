# v4 — « On retire la mémorisation, on garde la connaissance » (cible delta)

## Pourquoi
- Leaderboard v3/k1_a10 = 0.0143 (meilleur équipe : 0.0231 = soumission plus ancienne, origine exacte inconnue ;
  sans doute un des fichiers de `output/` (T2/T5/T8…, oubli plus fort que k1_a10)). k1_a10 était le run qui oubliait le moins
  → le gold oublie BEAUCOUP plus que « l'original sur une image modifiée ». Le problème n°1 est le NIVEAU d'oubli.

## Idée
- Cible = logits de l'original sur l'image exacte, en retirant `delta` au logit de la classe mémorisée.
  - Les autres classes gardent leur ordre : ce sont les classes visuellement proches, celles que le gold
    choisirait à la place (= la « mauvaise classe » exigée par le score de prédiction).
  - Un seul réglage, `delta`, contrôle la force d'oubli.
  - Queue lourde naturelle : les images avec une 2e classe plausible basculent, les autres restent justes.
- Reste identique à v2 : ancrage sur mosaïques (w_ret 3), vues augmentées avec la même cible, 5 époques, lr 5e-5.
- Nouveau suivi : `fit_vs_target` = la métrique officielle (T=2) en prenant la cible comme gold
  → dit si l'élève atteint la cible (1 = parfait). Ce n'est PAS le vrai score.
- `--diag` : histogramme des classes du forget set (images au hasard ou classes entières ?),
  marge top1-top2 de l'original, stats de la cible pour delta = 2…20.

## Outil : interp.py (CPU)
w = original + α × (oublié − original). Bouton continu pour affiner la force d'oubli sans réentraîner
(α > 1 = extrapolation). Esprit du rewinding SCRUB+R.

## Idée écartée pour l'instant : re-distillation (reset des derniers blocs + imitation sur vues modifiées)
- Le professeur a mémorisé aussi les vues modifiées (aug 1.0 : 87 % d'acc, conf 0.94) → l'élève réapprendrait
  le par-cœur, profil ≈ k1_a10 (mauvais). Risque retain élevé, pas de bouton de réglage. Possible v5.

## Diag (job 20261121-175912, GPU 1)
- Classes (pseudo-labels) : **45/100 classes présentes**, 55 absentes. Par classe : min 46, médiane 66, max 130.
- **9 classes plafonnent à 129–130 images** : 0, 6, 7, 8, 29, 30, 31, 33, 38 (≈ 1 170 images, 1/3 du forget set).
  Un plafond identique suggère que ce sont des **classes entières** (tout le train de la classe) → le gold ne les a
  jamais vues : ≈ 0 % d'acc dessus, il répond par des classes voisines. Les 36 autres classes : sous-ensemble partiel
  (46–104 images) → le gold les connaît, comportement « image de test ». (Hypothèse : 130 = taille d'une classe.)
- Marge de mémorisation top1-top2 : moyenne 12.2 logits (quantiles 10/50/90 % : 8.4 / 12.2 / 16.2).
- Cible delta (sans entraînement), T=1 : δ=8 → acc 0.93 · δ=10 → 0.77 · δ=12 → 0.52 · δ=14 → 0.28 · δ=16 → 0.11 · δ=20 → 0.01.
- → Un δ unique est un compromis : il faudrait δ grand sur les 9 classes entières et δ modéré sur le reste (piste v5).

## Runs (run_seq.sh, file GPU 1)
| run | delta | cible | clean loss ± std | acc | fit_vs_target | KL mosaïques | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|---|---|---|
| d4 | 4 | acc 0.997 | 0.02 ± 0.26 | 0.994 | 0.28 | 0.010 | 0.94 | |
| d8 | 8 | acc 0.925 | 0.25 ± 0.87 | 0.917 | 0.55 | 0.030 | 0.88 | |
| d12 | 12 | acc 0.517 | 0.68 ± 0.84 | 0.737 | 0.71 | 0.094 | 0.81 | **0.043** (21/11 ~18h15, nouveau meilleur) |
| d16 | 16 | acc 0.111 | 1.13 ± 0.90 | 0.601 | 0.51 | 0.146 | 0.77 | **0.0453** (21/11 ~18h25) |

Lecture : pour la 1re fois on descend sous 95 % d'acc sur le forget set (zone où doit se trouver le gold, ≤ 67 %).
Mais pour δ fort l'élève n'atteint pas la cible (d16 : vise 11 %, obtient 60 %) : l'ancrage sur des mosaïques
faites avec les images du forget set « rappelle » les classes. La dérive mosaïques monte avec δ.
