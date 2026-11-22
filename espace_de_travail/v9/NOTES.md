# v9 — Professeur cohérent (original + 9 classes entières éteintes)

## Pourquoi
- β est réglé (plateau 6–8, record 0.1381 avec v5/f20_p8 + β6). Tous les modèles existants plafonnent à ≈ 0.13.
- Jusqu'ici : on entraîne à oublier en imitant l'original sur les mosaïques (qui « rappelle » les 9 classes), puis on
  éteint après coup. Les deux étapes se contredisent.

## Idée
- Professeur = original avec les logits des 9 classes entières baissés de β=7 PARTOUT (`--teacher_beta 7`).
  L'élève démarre déjà éteint (`--init_shift`) et imite ce professeur sur les mosaïques et les vues modifiées.
- Sur les images à oublier : cible = professeur, logit de la vraie classe baissé de δ (20 pour les classes entières,
  8 pour les partielles).
- Variantes testées séparément (un facteur à la fois) :
  - `--delta_slope 8` : oubli proportionnel à la part de la classe retirée (46 à 104 images sur ~130) ;
  - `--aug_train 1.0` : copies d'entraînement plus modifiées (robustesse aux copies cachées du serveur) ;
  - `--k_aug 8` : cible moyennée sur 8 vues modifiées (le gold est une moyenne de plusieurs modèles).
- 1er essai annulé après 1 min (δ_full 8 : les classes entières restaient à 38 % dans la cible, car leur 2e choix est
  souvent une autre classe entière). Relancé avec δ_full 20.

## Runs (run_seq.sh, GPU 1, job 20261121-192358) + cache pour le simulateur (`sim.py`)
| run | variante | acc entières | acc partielles | accord mosaïques | simulé | Leaderboard |
|---|---|---|---|---|---|---|
| c7_f20_p8 | base | 0.046 | 0.940 | 0.766 | < 0.12 | **0.0833** (21/11 ~20h) |
| c7_f20_p8_s8 | + proportionnel | 0.044 | 0.933 | 0.765 | < 0.12 | |
| c7_f20_p8_a10 | + copies plus modifiées | 0.067 | 0.951 | 0.765 | < 0.12 | |
| c7_f20_k8_d4 | + cible moyennée, δ 4 | 0.068 | 0.972 | 0.766 | < 0.12 | |
| c7_f20_k8_d6 | + cible moyennée, δ 6 | 0.069 | 0.935 | 0.750 | 0.121 (avec β+2) | **0.1242** (21/11 ~19h50, < record 0.1387) |

Référence : v5/f20_p8 + β8 → 0.009 / 0.944 / 0.736. Record leaderboard : f20_p8 + β7 = **0.1387** (b6 0.1381, b8 0.1348).
Lecture locale : dérive un peu plus faible que la référence, classes entières un peu moins éteintes (4–7 %).
Le simulateur classe les v9 sous les modèles existants, mais il est peu fiable hors de la famille sur laquelle il est calé
(il a surestimé r_b8_e5_b2) → c'est le leaderboard qui tranche.
