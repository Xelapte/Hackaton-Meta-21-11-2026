# v56 — Indice « temps d'oubli » (SSFT, Maini et al., NeurIPS 2022) pour repérer les images que le gold rate

Idée trouvée par la recherche bibliographique du 22/11 à 8h30. On continue d'entraîner l'original sur une partie A des
images à oublier (20 par classe lourde, la moitié des autres : la situation du gold) et on observe à quelle vitesse il
oublie les autres images (B). Vite oubliée = image qui ne tenait que par le par-cœur. 8 tirages de A, 8 époques chacun.
Indice par image = marge moyenne de la bonne classe quand l'image est dans B. Troisième famille d'indices, après V
(vues modifiées) et T (profondeur / modèles dé-mémorisés). Résultats : voir v47/NOTES.md (listes `lourd_*`).

## Résultat (22/11, 9h08) : indice inutilisable tel quel
8 époques d'entraînement sur la partie A : l'original répond toujours juste sur 100 % des images de B (classes lourdes
comprises). Explication simple : A est déjà apprise par cœur (loss ≈ 0), donc il n'y a presque pas de gradient et rien ne
bouge. Le « temps d'oubli » demande une seconde partie de données NOUVELLES pour le modèle. Piste abandonnée faute de
temps ; l'équivalent utile est fait dans v57 (modèles ré-appris sur des images fabriquées, jamais sur les images à oublier).
