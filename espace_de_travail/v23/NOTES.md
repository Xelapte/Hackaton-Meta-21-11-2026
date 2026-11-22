# v23 — « Gold croisé » (testé dans la maquette réaliste) : piste fermée

## Idée
Entraîner nous-mêmes des modèles qui n'ont jamais vu chaque image à oublier (validation croisée en 5 paquets, classes
entières jamais montrées), prendre leurs réponses comme cible, les distiller dans l'original, puis extinction β.

## Résultat (21/11 ~23h20) : moins bien que la recette du record
- Qualité des cibles (écart au vrai gold, plus petit = mieux) : original 1.87 · cible du record 0.73 · gold croisé 1.81
  (1.36 au mieux en changeant la température) · un autre vrai gold 0.05.
- Score après distillation : 0.17 au mieux, contre 0.244 pour le record. Mélange avec la cible du record : 0.223.
- Raison : nos modèles croisés ne voient QUE les images à oublier (20 classes dans la maquette, 45 en vrai). Un vrai gold
  connaît toutes les classes et range souvent ses erreurs dans des classes absentes du forget set ; nos modèles croisés
  ne peuvent pas répondre ces classes-là. Leur « mauvaise réponse » est donc fausse.
- Enseignement utile : notre cible actuelle ne tombe d'accord avec le gold que lorsque le gold a JUSTE. Quand il se
  trompe, on ne trouve presque jamais sa mauvaise classe. C'est là qu'est la marge.
