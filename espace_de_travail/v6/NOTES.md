# v6 — Baisser le biais des 9 classes supprimées en entier (head shift) + évaluation automatique

## Pourquoi
- Leaderboard : v3/k1_a10 0.0143 → v4/d12 **0.043** → v4/d16 **0.0453**. Oublier plus a payé (×3), mais
  d12 → d16 ne gagne presque plus : oublier « uniformément plus » plafonne.
- v5 : les 9 classes entières résistent à l'entraînement (cible ≈ 0 % d'acc, obtenu 28–46 %).
- Raisonnement sur la métrique (loss à T=2, mean + std) : le gold n'a jamais appris ces 9 classes → sur leurs
  1 169 images sa loss est très élevée, sur les autres elle est faible → distribution **bimodale**, std énorme.
  Tant que nos modèles reconnaissent encore ces classes, on ne peut pas reproduire cette forme.

## Idée
- Le gold ne prédit (presque) jamais une classe qu'il n'a jamais vue, sur AUCUNE image.
  → on baisse directement le biais de ces 9 classes dans la dernière couche (`head.bias[c] -= β`).
- Effets :
  - sur une image de ces classes : le modèle répond sa 2e classe préférée (classe voisine), comme le gold ;
  - sur les autres images : ces classes avaient déjà une proba faible → presque aucun changement (retain préservé,
    surtout si le jeu de contrôle caché ne contient pas ces 9 classes, puisqu'elles sont retirées des données).
- Aucun entraînement : `head_shift.py` (CPU, secondes). Se combine avec n'importe quel modèle (d12, d16, v5…).
- Pour le jury : « une classe jamais vue n'est jamais prédite : on l'éteint dans la dernière couche ».
- Risque : si la validation cachée (seuil 70 %) contient ces 9 classes, on perd ~9 % des classes. Le gold les perd
  aussi, et il passe forcément le seuil.

## Outils
- `head_shift.py <modele.pt> <beta> <sortie.pt>` : crée le fichier à soumettre.
- `eval.py` / `run_eval.sh` : classement automatique (job GPU ~2-3 min) de plusieurs modèles × plusieurs β :
  acc / loss T=2 (tout, classes entières, partielles, vues modifiées) + dérive mosaïques (avec et sans les
  images des classes entières). Résultats dans `logs/eval.log` et `logs/eval.jsonl`.

## Résultats eval (job 20261121-182153, ~40 s) — extrait
| modèle | β | acc tout | acc entières | acc partielles | loss T2 ± std | KL mos (sans cl. entières) |
|---|---|---|---|---|---|---|
| original | 0 / 8 / 16 | 1.00 / 0.99 / 0.76 | 1.00 / 0.97 / 0.29 | 1.00 | | 0 / 0.28 / 0.68 |
| v4/d12 (LB 0.043) | 0 / 4 / 8 | 0.74 / 0.63 / 0.54 | 0.75 / 0.38 / 0.11 | 0.73–0.76 | 1.18±0.74 → 2.08±1.68 (β8) | 0.10 / 0.20 / 0.39 |
| v4/d16 (LB 0.0453) | 0 / 4 / 8 | 0.60 / 0.48 / 0.44 | 0.61 / 0.16 / 0.02 | 0.60–0.65 | | 0.15 / 0.25 / 0.44 |
| v5/f20_p8 | 0 / 4 / 8 | 0.77 / 0.67 / 0.63 | 0.42 / 0.11 / **0.01** | **0.94** | 0.94±0.83 → **2.05±2.30** (β8) | 0.08 / 0.19 / 0.38 |
| v5/f20_p8_wr1 | 0 / 4 / 8 | 0.72 / 0.64 / 0.63 | 0.28 / 0.05 / 0.00 | 0.94 | | 0.11 / 0.22 / 0.41 |

Lecture :
- Sur les modèles entraînés, β = 8 suffit à éteindre les classes entières (≈ 0–2 %) ; sur l'original il faudrait β > 16
  (marge de mémorisation ≈ 12 logits).
- f20_p8 + β8 donne enfin le profil « bimodal » visé : classes entières ≈ 0 %, partielles 94 %, std de loss 2.3.
- **Coût retain plus fort que prévu** : même sur des mosaïques sans ces classes, KL monte (0.08 → 0.38 à β8),
  car à T=2 ces 9 classes gardent une part de proba non négligeable sur des images ambiguës. Les mosaïques sont
  plus ambiguës que de vraies images (maxprob 0.71) → coût réel probablement plus faible, mais inconnu.

## Soumissions
| fichier | construction | Leaderboard |
|---|---|---|
| subs/f20_p8_b8.pt | v5/f20_p8 + head_shift β=8 | **0.1348** (21/11 ~18h40, ×3 vs d16, 1er du leaderboard) |

## Candidats prêts (21/11 ~19h, tous vérifiés strict=True / float32 / finis)
Règle : le MEILLEUR score public est retenu pour le classement final → chaque record nous engage (seuil 70 % !).
| fichier | bouton testé | acc entières / partielles | accord mosaïques | risque seuil 70 % | Leaderboard |
|---|---|---|---|---|---|
| f20_p8_b12.pt | β plus fort (12) | 0.000 / 0.944 | 0.73 | = b8 (mêmes prédictions) | **0.0592** (~19h05) → β=12 dépasse nettement le gold |
| f20_p8_b6.pt | β plus doux (6) | — | — | ≤ b8 | |
| f20_p10_b8.pt | autres catégories : 88 % | — | — | ≈ b8 | |
| d12_b8.pt | autres catégories : 76 % | 0.109 / 0.758 | 0.72 | un peu plus élevé | |
| d16_b8.pt | autres catégories : 65 % | 0.022 / 0.645 | 0.68 | le plus élevé → en dernier | |
- f20_p8_b6.pt : **0.1381** (21/11 ~19h20, nouveau meilleur). b6 ≈ b8 → plateau β 6–8 (voir v8/NOTES.md, simulateur).
- f20_p8_b7.pt : **0.1387** (21/11 ~19h35, record). Plateau β confirmé : b6 0.1381 · b7 0.1387 · b8 0.1348 · b12 0.0592.
- f20_p10_b7.pt : **0.1412** (21/11 ~20h30, record). Bouton « oubli des autres catégories » : δ_part 8 (94 %) → 0.1387 ; δ_part 10 (88 %) → 0.1412. Gain faible (+2 %), direction à confirmer avec δ_part 12/14.
