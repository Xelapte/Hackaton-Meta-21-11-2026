# v53 — Professeurs « dé-mémorisés » supplémentaires (pour repérer les images fragiles)

Même recette que v34 (`redistill.py`) : on remet à zéro les derniers blocs de l'original, puis il réapprend à imiter
l'original uniquement sur des photos extérieures (COCO). Il n'a donc plus le par-cœur des images à oublier.
Usage ici : ces modèles ne sont PAS soumis ; on regarde sur quelles images à oublier ils hésitent ou se trompent
(indice de fragilité indépendant de celui des « vues modifiées »). v47/hb_prof a montré que cet indice vaut l'autre (0.2858).
| run | blocs remis à zéro | graine |
|---|---|---|
| r2s1 | 2 | 1 |
| r3 | 3 | 2 |
| r2s3 | 2 (lr 3e-4) | 3 |

## Professeurs « dans la distribution » (8h40)
Limite des professeurs précédents : ils réapprennent l'original uniquement sur des photos extérieures (COCO), très
différentes de nos images. Nouveaux : 60 % de chaque lot = vues TRÈS modifiées des images à oublier (force 1,0 à 1,75) et
collages de 2 images — donc le bon type d'images, mais jamais l'image exacte (`--self_frac 0.6`).
| run | blocs remis à zéro | vues |
|---|---|---|
| in2 | 2 | 1,0 à 1,75 |
| in3 | 3 | 1,0 à 1,75 |
| in2h | 2 | 1,25 à 2,0 |
