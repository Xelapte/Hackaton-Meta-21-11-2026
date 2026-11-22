# v32 — Effacer puis réparer sur le VRAI modèle, avec des images synthétiques

## Idée
Référence officielle « Zero-Shot Machine Unlearning » : fabriquer des images de remplacement à partir du modèle seul.
1. `gen.py` : 4 000 images (40 par classe) obtenues en optimisant du bruit jusqu'à ce que l'original y voie la classe.
   L'original reconnaît la classe voulue dans 100 % d'entre elles. ~3 min par lot de 1 000 sur un A100.
2. `er.py` : on abîme les 4 derniers blocs (bruit, ou remise à zéro de 2 blocs), puis l'élève réapprend à imiter
   l'original sur 64 images synthétiques + 32 mosaïques par lot. Classes 0 et 38 baissées de 7 chez le professeur.

## Résultats locaux (22/11 ~1h00) : non soumis
| variante | bonnes réponses sur le forget set (classes partielles) | accord mosaïques avec l'original |
|---|---|---|
| record (référence) | 0.92 | 0.75 |
| bruit 0.5 | 0.998 | 0.71 |
| bruit 1.0 | 0.98 | 0.61 |
| bruit 2.0 | ≈ 0.60 | ≈ 0.42 |
| 2 blocs remis à zéro | ≈ 0.83 | ≈ 0.52 |
- Bruit faible : le par-cœur n'est pas cassé. Bruit fort / remise à zéro : le par-cœur est cassé, mais le modèle n'imite
  plus l'original que sur les images synthétiques (97 %), pas sur des images réalistes.
- Les images tirées du modèle ne ressemblent pas assez à de vraies images pour lui rendre son savoir général.
- Non soumis : trop éloigné de l'original (risque sur le seuil des 70 %, et le meilleur score public est retenu).
- La maquette (v29) annonçait « aucun gain » : confirmé. Coût : 15 min de GPU, aucun créneau de soumission.
