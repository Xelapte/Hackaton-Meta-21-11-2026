# v55 — Variantes « fraîches » des images à oublier, fabriquées par diffusion (SD-Turbo, image vers image)

## Pourquoi (8h45, 22/11)
Il nous manque depuis le début des images fraîches de la même distribution. Toutes nos estimations de « ce que répondrait
un modèle qui n'a jamais vu l'image » viennent des images à oublier elles-mêmes (recadrages, couleurs : même scène).
Les images du jeu de données sont générées par IA : on peut en refabriquer du même style.

## Méthode
- Modèle : `stabilityai/sd-turbo` (licence Stability AI Community, usage recherche), téléchargé dans `ext/` avec l'accord
  de E. ; bibliothèque `diffusers` installée par E. dans `pylib_gen/` (hors du `.venv` de l'équipe).
- `gen_var.py` : chaque image (224 → 512) est « redessinée » en partie (4 pas, force 0,5 à 0,6, texte neutre « cinematic
  photograph… »), puis ramenée à 224. La scène et le style restent, les détails changent.
- Essai sur 200 images : force 0,5 → l'original reconnaît la bonne classe sur 89,5 % des variantes (vraie classe 0,80 à
  T=2 contre 0,99 sur l'image exacte) ; force 0,75 → 23,5 % seulement (l'objet disparaît) : trop fort sans nom de classe.

## Usages prévus
1. Nouvel indice de fragilité (G) : marge de l'original sur les variantes, à combiner avec V (vues) et T (profondeur).
2. Ancrage du retain sur les variantes (l'élève doit y rester l'original) sans toucher à l'image exacte.
