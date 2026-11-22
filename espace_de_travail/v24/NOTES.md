# v24 — Quelle transformation « démémorise » le mieux ? (diagnostic dans la maquette) : aucune

- 23 transformations testées (zoom, basse résolution, flou, rotation, gris, effacement, morceaux mélangés, fondu, combinaisons).
- Écart au vrai gold sur les classes partielles (plus petit = mieux) : un autre gold 0.05 ; meilleure cible trouvée 0.60 ;
  cible du record 0.80 ; original brut 1.90.
- Aucune transformation ne fait répondre l'original comme un gold : soit il reconnaît encore l'image, soit il se trompe
  autrement que le gold (accord avec le gold ≤ 0.45, contre 0.78 entre deux gold).
- Désaccord avec le leaderboard réel sur un point : la maquette préfère l'image exacte aux 16 vues, alors qu'en vrai les
  vues ont rapporté +5 % puis +3 %. La maquette n'est donc pas fiable pour régler la construction des cibles.
- Conclusion : la « mauvaise classe » du gold n'est pas lisible dans l'original par simple transformation des images.
