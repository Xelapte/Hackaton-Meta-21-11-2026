# v16 — Exploration large autour du record

## Pourquoi
- Le « un bouton à la fois » est lent. Les deux boutons principaux sont calés (extinction β≈7 ; 88–94 % d'acc sur les
  classes partielles) et la cible moyennée donne le record (v11/f20_k8_d6_b7 = **0.1462**).
- Réglages jamais variés : ancrage (w_ret), vitesse/durée (lr, époques), température de la cible (tau), poids et force
  des copies modifiées (w_aug, aug_train), couches libres (train_blocks).

## Méthode
- 8 modèles variés, plusieurs réglages changent à la fois (run_seq.sh). Extinction β=7 sur tous.
- Lecture : un score isolé ne dit pas quel réglage compte ; on ne retient un réglage que s'il revient chez plusieurs
  bons modèles (garde-fou contre le surajustement au leaderboard public).
- En plus, sans GPU : mélanges de poids original ↔ record (`interp.py`) : α=0.8 (80 % du chemin) et α=1.2 (20 % plus loin).

## Runs
| run | réglages changés | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| e1 | w_ret 1 | 0.469 | 0.926 | 0.73 | **0.0848** (23h31) : bien pire que v18/k16_wr1 (0.1528), qui ne diffère que par 16 vues au lieu de 8 |
| e2 | w_ret 6, 8 époques | 0.534 | 0.892 | 0.81 | **0.0959** (22h07) : ancrage fort + long = nettement moins bien |
| e3 | tau 1.5, δ_part 4 | 0.481 | 0.969 | 0.82 | **0.1324** (21h45 ; resoumis par erreur à 21h55, même score) |
| e4 | lr 1e-4, 3 époques, w_aug 2 | 0.528 | 0.905 | 0.76 | |
| e5 | 4 derniers blocs, lr 1e-4, 8 époques | 0.489 | 0.932 | 0.79 | |
| e6 | 16 vues, aug 1.1, δ_part 5, w_ret 2, aug_train 0.75 | 0.523 | 0.928 | 0.77 | |
| e7 | lr 2e-5, 10 époques, w_aug 0.5, δ_part 7 | 0.554 | 0.855 | 0.78 | |
| e8 | 16 vues, δ_part 5, tau 1.25, w_ret 1, 6 derniers blocs | | | | |
| rec_a0.8 | mélange α=0.8 du record | — | — | — | |
| rec_a1.2 | mélange α=1.2 de v11/f20_k8_d6_b7 (0.1462) | — | — | — | **0.1202** (21h33, soumis par Claude) : aller plus loin fait baisser |

- e8 : ÉCHEC à l'enregistrement (« Disk quota exceeded » côté compte de la file, 21h20).
