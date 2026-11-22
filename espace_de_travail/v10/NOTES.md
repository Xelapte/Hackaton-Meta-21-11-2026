# v10 — Incliner la « mauvaise réponse » vers les classes ressemblantes

## Pourquoi
- Plateau à ≈ 0.12–0.14 (record v6/f20_p8_b7 = 0.1387). v9 (cible moyennée) = 0.1242 : pas mieux.
- Littérature (thèse arXiv 2512.06993, méthode TRW) : un modèle réentraîné sans une classe répond surtout les classes
  qui lui ressemblent. Éteindre la classe ne suffit pas, il faut reproduire ce motif (= notre note « prédiction »).

## Idée
- Même recette que le record (v5/f20_p8 : δ_full 20, δ_part 8, puis extinction β=7).
- Nouveau : dans la cible des images des 9 classes entières, on ajoute `tilt × ressemblance(classe retirée, classe k)`
  au logit de chaque classe restante k. Ressemblance = cosinus entre les vecteurs de poids de la dernière couche.
- Vérification : les voisines trouvées sont plausibles (classes d'indices proches) : 0→2,1 · 29→32,28,27 · 38→36,34,37.
  Les cosinus sont petits (max 0.30) → on teste tilt 10 / 20 / 40.
- Différence avec TRW : pas de retain set (on garde nos mosaïques) ; formules exactes du papier non vérifiées mot à mot.

## Runs (run_seq.sh)
| run | tilt | acc entières | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| f20_p8_t10_b7 | 10 | 0.388 avant β | 0.945 | 0.82 avant β | |
| f20_p8_t20_b7 | 20 | 0.358 avant β | 0.945 | 0.81 avant β | |
| f20_p8_t40_b7 | 40 | 0.249 avant β | 0.945 | 0.80 avant β | **0.121** (21/11 ~20h20, < record 0.1387 sans tilt) |

## Lecture du papier (papers/trw.pdf, sections 5.1.4 et C.2.2, lues mot à mot le 21/11 ~20h15)
- Cible TRW : q*(y|x) ∝ p̃(y|x) · exp(β·s_y), avec p̃ = proba de l'original renormalisée sans la classe retirée, β = 10.
- Score s_y : cosinus entre les vecteurs de poids de la dernière couche (après projection PCA), **passé dans un softmax
  de température 0.01**. s_y est donc ≈ 1 pour LA classe la plus ressemblante et ≈ 0 ailleurs → en pratique : ≈ +10
  logits sur la voisine la plus proche (partagés en cas de quasi-égalité).
- Constat du papier : le modèle réentraîné range les images de la classe retirée dans la classe voisine « avec une
  certitude bien plus élevée » que ne le suggère l'original renormalisé.
- Notre v10 (tilt × cosinus brut) est plus étalée : t40 donne +12 sur la 1re voisine mais aussi +7 sur la 2e.
  → variante fidèle à tester ensuite (v12) : s = softmax(cos / 0.01), β = 10.
- Résultats locaux v10 (avant extinction β7) : plus le tilt est fort, moins les 9 classes sont reconnues (39 % → 36 % → 25 %).
