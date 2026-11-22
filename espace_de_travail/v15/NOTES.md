# v15 — Exploration de la cible moyennée

## Pourquoi
- Seule idée qui a nettement fait monter le score sur la bonne base : cible = moyenne des réponses de l'original sur
  8 vues modifiées (v11/f20_k8_d6_b7 = **0.1462**, contre 0.1387 sans moyenne). Testée avec un seul réglage.
- Interprétation : le gold est une moyenne de plusieurs modèles → ses probabilités sont « lissées » ; moyenner sur des
  vues modifiées lisse aussi les nôtres, y compris le choix de la mauvaise classe.

## Runs (un élément change à la fois autour du record : k_aug 8, aug 1.0, δ_part 6, δ_full 20, puis β=7)
Attention : changer la force des vues change aussi un peu le niveau d'oubli (la vraie classe domine moins sur des vues
plus modifiées) → regarder l'acc partielles avant de conclure.
| run | changement | cible : acc partielles | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|---|
| f20_k16_d6_b7 | 16 vues | 0.872 | 0.461 | 0.934 | 0.81 | **0.1506** (21/11 ~21h20, RECORD) |
| f20_k8_a125_d6_b7 | vues plus modifiées (1.25) | 0.631 | 0.512 | 0.847 | 0.79 | |
| f20_k8_a075_d6_b7 | vues moins modifiées (0.75) | 0.957 | 0.446 | 0.977 | 0.83 | |
