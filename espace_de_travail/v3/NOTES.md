# v3 — cible « à queue lourde » (code = v2/unlearn.py, seuls les arguments changent)

## Idée
- Hypothèse : le gold se comporte sur une image du forget set comme l'original sur une image modifiée
  qu'il n'a jamais vue : confiant le plus souvent, très faux parfois (distribution de loss à queue lourde).
- Cible par image = prédiction de l'original sur 1 à 4 vues fortement modifiées, SANS tau (tau = 1).
  Moyenner beaucoup de vues (v1/v2) écrase la queue → peu de vues.
- Reste identique à v2 : w_ret 3, aug_train 0.5, lr 5e-5, 5 époques, bs 48.

## Runs — métriques à la dernière époque
| run | k_aug | aug | cible (loss ± std, acc, conf) | clean loss ± std | acc | conf | KL mosaïques T2 | accord mosaïques |
|---|---|---|---|---|---|---|---|---|
| k1_a10 | 1 | 1.0 | 0.59 ± 1.81, 0.87, 0.94 | 0.13 ± 0.41 | 0.951 | 0.94 | 0.046 | 0.87 |
| k2_a10 | 2 | 1.0 | 0.24 ± 0.59, 0.95, 0.86 | 0.11 ± 0.23 | 0.975 | 0.92 | 0.052 | 0.87 |
| k1_a12 | 1 | 1.25 | 1.47 ± 2.96, 0.72, 0.89 | 0.29 ± 0.45 | 0.930 | 0.82 | 0.084 | 0.81 |
| k4_a12 | 4 | 1.25 | 0.42 ± 0.50, 0.92, 0.73 | 0.30 ± 0.36 | 0.958 | 0.79 | 0.087 | 0.83 |

## Lecture
- Retain : bien meilleur que v2 (KL mosaïques 0.05–0.09 contre 0.12–0.21).
- Forget : la cible a la bonne forme, mais l'élève ne l'atteint pas (std plafonne ≈ 0.4, acc 0.93–0.95
  au lieu de 0.72–0.87). Le plateau est atteint dès l'époque 3–4 : l'ancrage mosaïques (w_ret 3) freine
  probablement l'oubli → piste v4 : w_ret plus faible, ou ancrage limité aux premiers blocs.
- Moyenner 2–4 vues réduit la queue (k2, k4) → contraire à l'hypothèse.
- Leaderboard (21/11 ~17h50) : **k1_a10 = 0.0143** (public split). Meilleur précédent de l'équipe : 0.0231 (fichier à identifier).
  → Bien en dessous de l'estimation (~0.2) : le gold oublie probablement BEAUCOUP plus que notre hypothèse
  (k1_a10 est le run qui oublie le moins). Direction : plus d'oubli.
