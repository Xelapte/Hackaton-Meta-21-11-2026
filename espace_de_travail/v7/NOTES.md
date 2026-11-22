# v7 — Réparation du retain après l'extinction des 9 classes entières

## Pourquoi
- v6/f20_p8_b8 = **0.1348** (1er du leaderboard). Mais l'extinction par le biais agit sur TOUTES les images :
  même sur des mosaïques sans ces classes, la KL vers l'original monte de 0.08 à 0.38 → on perd du retain.

## Idée
- Point de départ = le modèle v6 (`--init`), qui a déjà oublié.
- Cible sur le forget set = ses propres prédictions (`--target init`) → on GARDE l'oubli (classes éteintes).
- Ancrage = imiter l'original sur des mosaïques construites **sans** les images des 9 classes
  (`--mos_exclude_full`) → le modèle redevient comme l'original sur ce qui n'est pas à oublier.
- Le modèle apprend donc à n'éteindre ces classes **que là où il faut**, au lieu de partout.
- Esprit SCRUB : les dernières époques « min » servent à restaurer le retain.

## Runs (run_seq.sh) + eval automatique à la fin (eval.py, avec b8/b12 comme référence)
| run | départ | époques | lr | w_ret | acc entières | acc partielles | KL mos (sans cl. entières) | Leaderboard |
|---|---|---|---|---|---|---|---|---|
| r_b8 | f20_p8_b8 | 3 | 2e-5 | 3 | 0.096 | 0.946 | 0.114 (all) | **0.1247** (< 0.1348) |
| r_b8_wr6 | f20_p8_b8 | 3 | 2e-5 | 6 | 0.139 | 0.953 | 0.104 (all) | |
| r_b8_e5 | f20_p8_b8 | 5 | 3e-5 | 3 | 0.047 | 0.937 | 0.085 (all) | |
| r_b12 | f20_p8_b12 | 3 | 2e-5 | 3 | 0.050 | 0.948 | 0.209 (all) | |

Référence v6 (même mesure, mosaïques toutes images) : b8 → acc entières 0.009, KL mos 0.479 ; b12 → 0.000, 0.744.
Note : l'éval automatique de fin de job a échoué (chemins relatifs résolus après le chdir de unlearn.py) ;
métriques ci-dessus = celles de l'entraînement.

Lecture leaderboard : r_b8 (0.1247) < b8 (0.1348) alors que la dérive mosaïques est divisée par 4.
→ La réparation a « rallumé » un peu les classes entières (1 % → 10 %) et ça coûte plus que le retain gagné.
→ La dérive mesurée sur mosaïques surestime le coût retain réel. Le terme forget domine : il faut éteindre
  davantage (b12) plutôt que réparer.
