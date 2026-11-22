# v33 — Se tromper sur les images difficiles plutôt que sur des classes entières

## Pourquoi
- v27/v31 : le score dépend surtout du NOMBRE d'images sur lesquelles on se trompe franchement (optimum ≈ 650–780, soit
  ≈ 20 % du forget set), pas de leur classe. Éteindre des classes entières réglait « combien », pas « lesquelles ».
- Un gold se trompe sur les images réellement difficiles (ambiguës, peu typiques), pas sur des classes au hasard.

## Idée
- Difficulté d'une image = marge de sa vraie classe dans les réponses de l'original moyennées sur 16 vues modifiées.
- Les 20 % les plus difficiles : cible = vraie classe baissée de 25 → le modèle répond avec assurance sa 2e idée.
- Les 80 % restants : oubli modéré. Pas d'extinction globale (elle ne sait viser que des classes).
- Mosaïques d'ancrage construites sans les images difficiles ; poids renforcé sur ces images ; 8 époques.
- Point à surveiller : la loss sur les images difficiles doit monter vers ≈ 8 (niveau obtenu par l'extinction β=7).

## Runs
| run | réglages | acc / loss images difficiles | acc autres | accord mosaïques | Leaderboard |
|---|---|---|---|---|---|
| h20_w4 | poids 4, δ_part 3 | 0.003 / 3.77 | 1.000 | 0.82 | **0.0837** (1h32) : moitié du record 0.1660 |
| h20_w10 | poids 10, δ_part 3 | 0.003 / 3.94 | 0.989 | 0.77 | |
| h20_w10b | poids 10, δ_part 6, lr 1e-4 | | | | |


Lecture locale (1h20) : 700 images difficiles, dont les classes les plus touchées sont 7, 31, 30, 33, 6, 0, 8 (celles qu'on
éteignait : elles sont surtout difficiles parce qu'elles se ressemblent). Le modèle se trompe bien dessus (0,3 % de bonnes
réponses) et reste proche de l'original, mais pas assez franchement : loss 3.8–3.9 pour ≈ 8 visé, même avec un poids ×10.
h20_w10b a échoué (voir son journal). h20_w4 soumis une fois pour savoir si le choix « par image » a un intérêt.
Verdict (1h35) : se tromper « par image difficile » sans extinction ne marche pas en l'état (0.0837). Piste mise de côté.