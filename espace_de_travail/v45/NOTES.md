# v45 — Outils d'analyse (aucun modèle soumis) : lire le leaderboard et choisir les réglages sans entraînement

| fichier | rôle |
|---|---|
| `diag_classes.py`, `img/*.jpg` | planches d'images par classe (familles de classes sœurs), solidité de chaque classe |
| `eval_models.py`, `eval_vues.py`, `bilan.sh` | met en cache les sorties d'un modèle (image exacte, 4 copies légères, vues fortes) |
| `fit2.py` | **modèle du leaderboard** : ajuste la moyenne/écart-type de loss du gold (MU ≈ 1,4, SIGMA ≈ 4,25) sur nos notes |
| `choix_beta.py`, `choix1.py`, `tableau.py`, `bilan_h.py` | pour un modèle en cache : meilleur biais, mean_std prédit, forme des réponses à T=2 |
| `implique_tout.py` | note = mean_std prédit × « reste » pour toutes les anciennes soumissions |
| `derive.py` | « retain local » : écart à l'original sur des vues très modifiées |
| `marges.py`, `feats.py`, `peu_exemples.py`, `fragile2.py`, `indices.pt` | indices de fragilité des images (vues, peu d'exemples, bruit, élagage, lecture anticipée) |
Les conclusions tirées de ces outils sont dans `JOURNAL.md` (découvertes 6 à 11) et `v43/NOTES.md`, `v47/NOTES.md`.
