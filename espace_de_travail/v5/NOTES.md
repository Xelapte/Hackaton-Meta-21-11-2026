# v5 — Oubli différencié : classes supprimées en entier vs classes partielles

## Pourquoi (diag v4)
- Forget set = 45 classes sur 100. **9 classes à 129–130 images** (0, 6, 7, 8, 29, 30, 31, 33, 38, ≈ 1 170 images) :
  plafond identique → probablement des **classes entières** retirées. 36 autres classes : 46–104 images (partielles).
- Le gold n'a jamais vu les 9 classes entières → ≈ 0 % d'acc dessus, il répond par des classes voisines.
  Sur les classes partielles, il connaît la classe → comportement « image de test » (souvent juste, parfois faux).
- Un δ unique (v4) ne peut pas reproduire ces deux comportements à la fois.

## Idée
- Même cible que v4 (logits de l'original, logit de la classe mémorisée baissé de δ), mais :
  - `--delta_full` (grand, ≈ 16–20 → ≈ 0–10 % d'acc) sur les images des classes entières ;
  - `--delta_part` (modéré, ≈ 8–10 → ≈ 75–90 % d'acc) sur les classes partielles.
- Classe « entière » = nb d'images (pseudo-labels) ≥ `--full_thr` (125).
- Option `--mos_full_delta` : les mosaïques sont faites avec les images du forget set, donc l'ancrage
  « imiter l'original sur les mosaïques » réinjecte les classes entières. Cette option les baisse aussi dans
  la cible des mosaïques. Risque : dérive sur le jeu de contrôle si celui-ci contient ces classes.
- Suivi par époque séparé : `clean_full` / `clean_part`.
- Pour le jury : « on a détecté dans les données que 9 classes étaient supprimées en entier, et on oublie
  différemment une classe entière et quelques images d'une classe ».

## Runs (grille ajustée après v4 : l élève n atteint pas les cibles fortes → option mosaïques et w_ret 1)
| run | δ_full | δ_part | mos | acc entières | acc partielles | fit | KL mosaïques | accord mos. | Leaderboard |
|---|---|---|---|---|---|---|---|---|---|
| f20_p8 | 20 | 8 | 0 | 0.421 | 0.942 | 0.36 | 0.085 | 0.82 | |
| f20_p10 | 20 | 10 | 0 | 0.457 | 0.876 | 0.39 | 0.099 | 0.79 | |
| f20_p8_mos4 | 20 | 8 | 4 | 0.382 | 0.945 | 0.37 | 0.201 | 0.75 | |
| f20_p8_wr1 (w_ret 1) | 20 | 8 | 0 | 0.280 | 0.936 | 0.40 | 0.121 | 0.78 | |

Lecture : les 9 classes entières ne sont PAS oubliées (cible ≈ 0.5 % d'acc, obtenu 28–46 %).
- w_ret 1 aide le plus (28 %). mos_full_delta 4 double la dérive mosaïques (0.20) pour presque rien → écarté.
- Cause probable : les mosaïques contiennent des images des classes entières → l'ancrage les « rappelle ».
  → piste v6 : construire les mosaïques SANS les images des classes entières (pas de conflit, cible mosaïques inchangée).
