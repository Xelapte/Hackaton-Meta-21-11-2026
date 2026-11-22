# v52 — Petites améliorations autour du record (recette v36/ext_wr1 + logits ÷ 0,8 + biais 0/38)

## Pourquoi (6h15, 22/11)
Le record (0.2455) vient d'une forme précise des réponses à T=2 (v45, par groupe) :
- classes « partielles » : assez sûres (vraie classe 0,65, voisines 0,18, sans rapport 0,17) ;
- lanternes et bateaux (classes presque vidées) : très hésitantes ENTRE SŒURS (vraie 0,23, voisines 0,47, sans rapport 0,31).
Rendre lanternes/bateaux plus sûrs (v47/c38 ÷ 0,8 : vraie 0,61) coûte 11 %. Logique : moins il reste d'images d'une classe
chez le gold, plus il hésite avec les classes sœurs. On teste ici des variantes qui vont dans ce sens.

## Runs (tous : 9 classes à δ 20, les autres δ 6, 16 vues, ancrage COCO 1)
| run | changement | Leaderboard (après ÷ τ et biais) |
|---|---|---|
| tm05 | queue de la cible lissée : 50 % image + 50 % moyenne de la classe (`--tail_mix 0.5`) | |
| tm00 | queue = moyenne de la classe seulement | |
| g1, g2 | mêmes réglages que le record, autres graines (pour une moyenne de poids de 3 modèles) | |
| k64 | 64 vues au lieu de 16 | |
| sl10, sl20 | oubli gradué selon la part retirée de chaque classe (`--delta_slope 10 / 20` : δ de 4,5 à 9 / de 3 à 12) | |
