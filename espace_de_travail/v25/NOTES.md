# v25 — Forme de la cible des classes entières (vrai modèle)

## Pourquoi
- Record : v18/k16_wr1_at025_b7 = **0.1552** (16 vues, ancrage 1, copies légères, puis β=7).
- Les maquettes v20–v24 montrent que notre famille de recettes plafonne et que la marge est dans la « mauvaise réponse ».
- Point jamais testé : les 9 classes entières se ressemblent entre elles (6/7/8 → toutes vers la 5 ; 29/30/31/33 → vers la 32).
  Notre cible ne baisse que la classe de l'image : une image de la classe 6 est donc rangée en partie dans 7 ou 8, que le
  gold ne prédit jamais non plus (l'extinction β=7 ne les baisse que de 7).

## Runs (un élément à la fois, puis β=7)
| run | changement | acc entières (avant β) | acc partielles | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| mfa_b7 | cible : toutes les classes entières baissées | 0.089 | 0.925 | 0.75 | mfa_b6 = **0.1169** (23h42) : nettement sous le record 0.1552 |
| tf05_b7 | cible des classes entières plus tranchée (T 0.5) | 0.298 | 0.927 | 0.75 | |
| tf2_b7 | cible des classes entières plus hésitante (T 2) | 0.392 | 0.925 | 0.75 | |
| wr03_at01_b7 | ancrage 0.3 + copies très légères (pousse la direction du record) | 0.246 | 0.917 | 0.64 | |


Lecture locale : avec `mfa`, les classes entières tombent à 9 % de bonnes réponses dès l'entraînement (32 % pour le record) ;
leur loss avant β vaut 3.1 au lieu de 1.7 → pour rester au même niveau d'extinction totale que le record, on soumet d'abord
`mfa_b6` (β=6) plutôt que β=7. `wr03_at01` dérive trop de l'original (accord 0.64) : non soumis (risque seuil 70 %).