# v2 — v1 sans double aplatissement + ancrage retain renforcé

## Idée (diff vs v1)
- Vues augmentées du forget set : même cible que l'image exacte (`--aug_target same`), plus original/tau.
- Augmentation d'entraînement plus légère (`--aug_train 0.5`), w_ret 1 → 3, lr 1e-5 → 5e-5, 3 → 5 époques.
- Cible = moyenne des softmax sur 16 vues (aug 1.0), aplatie par tau.

## Runs (job via file GPU 1, run_seq.sh) — métriques à la dernière époque
| run | tau | lr | cible (loss ± std, conf) | clean loss ± std | acc | conf | KL mosaïques T2 | accord mosaïques |
|---|---|---|---|---|---|---|---|---|
| t2 | 2.0 | 5e-5 | 0.64 ± 0.32, 0.55 | 0.59 ± 0.52 | 0.924 | 0.63 | 0.116 | 0.80 |
| t3 | 3.0 | 5e-5 | 1.22 ± 0.42, 0.32 | 1.12 ± 0.73 | 0.807 | 0.43 | 0.167 | 0.75 |
| t4 | 4.0 | 5e-5 | 1.75 ± 0.42, 0.19 | 1.76 ± 0.93 | 0.624 | 0.29 | 0.213 | 0.71 |
| t3_lr1e4 | 3.0 | 1e-4 | 1.22 ± 0.42, 0.32 | 1.24 ± 0.68 | 0.838 | 0.37 | 0.164 | 0.75 |

## Conclusion
- La cible moyennée + tau donne une loss UNIFORME (std ≈ 0.4) et une confiance faible partout.
  Un modèle « jamais vu » ressemble plutôt à : très confiant la plupart du temps, franchement faux parfois
  (original sur aug 1.0 : loss 0.60 ± 1.86, conf 0.94). → motive v3.
- Dérive sur les mosaïques élevée (accord 0.70–0.80) : risque côté retain / seuil 70 %.
- Leaderboard : —
