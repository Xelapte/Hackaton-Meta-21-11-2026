# Tâche 1 : localiser le tatouage dans l'image

Dossier officiel : `$OFFICIEL/Hackathon_Watermark_Localisation_Task/`. **Le jeu de données est à accès contrôlé** (`gated: manual`) : chaque compte Hugging Face demande l'accès sur la page du dataset, un organisateur valide, puis on relance `hackathon_setup.sh` avec `HF_TOKEN`. Contenu annoncé : `README.txt`, `starter_kit.zip`, les images. **Lisez le README dès qu'il est là et complétez cette page.**

## Le problème (d'après la présentation)
- Images 512×512, tatouées **sur une partie seulement**, par l'une de trois méthodes : **PixelSeal, TrustMark, StegaStamp**.
- Évaluation sur une **grille 128×128** (cases de 4×4 pixels) : 1 si au moins la moitié de la case est tatouée, sinon 0.
- **Score** : IoU de la classe « tatoué » et IoU de la classe « propre », moyennées par image (mIoU), puis moyenne sur toutes les images.
- **Données** : 1 530 images, à parts égales entre les 3 méthodes.
  - Entraînement : 510 images (170 par méthode), avec le **masque de vérité** (512×512 et 128×128), la **méthode** et le **message** inséré.
  - Test : 1 020 images, avec la **méthode** et le **message**, sans masque.

## Pistes
1. **Segmentation apprise** : un petit réseau (U-Net, encodeur pré-entraîné) qui prend l'image, éventuellement la méthode, et sort une carte 128×128. Perte BCE + Dice. Validation : 20 % des 510 images, score mIoU local identique à celui du serveur.
2. **Décodage local** (on connaît la méthode et le message de chaque image de test) : décoder le message par fenêtres et marquer « tatouées » les zones où les bits correspondent. Pour PixelSeal, l'extracteur produit une prédiction par pixel avant la moyenne finale : comparer ces prédictions au message connu donne directement une carte (bibliothèque videoseal : `setup/install_pixelseal.sh`, poids via `setup/download_all.sh $DIR` sans modèle de langage).
3. **Combiner** : utiliser la carte de décodage comme entrée supplémentaire du réseau de segmentation.

## Où ça tourne
Entraînement et inférence : dans la file du **GPU 2** (`bash slurm/soumettre.sh 2 <nom> --venv watermark python ...`). Si le kit officiel n'a pas de `requirements.txt` (pas de `.venv` créé), construisez le venv à la main dans le dossier de la tâche avec `uv venv -p 3.12 .venv` puis `uv pip install ...`.
