# v8 — Simulateur du score, calibré sur nos notes du leaderboard

## Idée
- On ne connaît pas le gold → on fabrique un « faux gold » réglable à partir de l'original :
  on retire `dp` au logit de la vraie classe (plus de par-cœur), `a` aux 9 classes entières, on divise par `tau`.
- On rejoue la formule officielle en local (forget sur images exactes + copies modifiées, retain sur mosaïques).
- On cherche les réglages du faux gold qui reproduisent nos vraies notes (`lb.json`), puis on classe tous les
  candidats (modèle × β). Aucun GPU : les réponses de 14 modèles sont enregistrées une fois (`cache.py`, job ~2 min).
- Rôle : **boussole / filtre**. Le leaderboard décide.

## Fichiers
- `cache.py` / `run_cache.sh` : enregistre les logits (images exactes, 2+2 vues modifiées, mosaïques) dans `cache/`.
- `lb.json` : nos notes réelles (à compléter après chaque soumission).
- `sim.py` : 1re version (loss à T=2). `sim2.py` : version retenue (teste loss à T=1 ou T=2).

## Résultats (21/11 ~19h20, 7 notes de calibration)
- Meilleur calibrage : **loss mean/std à T=1** (erreur ≈ 8–10 %, contre 10–13 % à T=2). Cohérent avec task_template.py :
  T=2 n'est écrit que pour les KL. (CONTEXTE.md disait « T=2 partout » : probablement faux pour la loss.)
- Test à l'aveugle réussi : avant de connaître la note, le simulateur donnait b6 ≈ 0.134 → réel **0.1381**.
- β : plateau 6–8, pic simulé à 7 (b5 0.125 · b6 0.138 · b7 0.148 · b8 0.140 · b9 0.122 · b12 0.059).
  → β est réglé ; plus rien d'important à gagner sur ce bouton.
- Parmi les modèles existants, tout plafonne à ≈ 0.14–0.16 simulé. Meilleur simulé : v7/r_b8_e5 + β2 (0.16, fourchette
  0.125–0.19) → fabriqué : `subs/r_b8_e5_b2.pt`.
- **Facteur inexpliqué ×0.25** : le simulateur doit multiplier tous ses scores par ≈ 0.25 pour coller au réel, pour TOUS
  les modèles. Il y a donc une partie du comportement du gold qu'aucun de nos modèles ne capte (hypothèses : la
  « mauvaise classe » image par image, ou les copies modifiées du serveur). C'est là qu'est la marge (jusqu'à ×4).

## Soumissions issues du simulateur
| fichier | simulé | réel |
|---|---|---|
| v6/subs/f20_p8_b6.pt | 0.134 (prédit avant) | **0.1381** |
| v8/subs/r_b8_e5_b2.pt | 0.16 (0.125–0.19) | |
| v6/subs/f20_p8_b7.pt | 0.148 | |

## Mise à jour 21/11 ~19h30
- `v8/subs/r_b8_e5_b2.pt` : simulé 0.16 (0.125–0.19) → réel **0.1198**. Le simulateur a surestimé ce candidat :
  il sur-récompense les modèles « réparés » (faible dérive sur mosaïques). 2e fois que la réparation perd (v7 r_b8 aussi).
- Recalibré avec 8 notes : tous les modèles existants sont à ≈ 0.13 ± 0.02 simulé → le simulateur ne sait plus les
  départager. Conclusion : on est sur un plateau avec cette famille de modèles ; le record reste f20_p8 + β6 (0.1381).
- Leçon : le simulateur est fiable pour interpoler un bouton déjà exploré (β), pas pour extrapoler à une autre famille.
