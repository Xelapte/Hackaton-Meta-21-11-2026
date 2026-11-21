# Tâche 1 : localiser le tatouage dans l'image

Dossier officiel : `$OFFICIEL/Hackathon_Watermark_Localisation_Task/` (venv : `--venv watermark`, construit à la main, voir plus bas). Fichiers fournis : `README.txt`, `starter_kit.zip`, `train.zip` (190 Mo), `test.zip` (379 Mo).

## Le problème
- Images 512×512, tatouées **sur une partie seulement**, par l'une de trois méthodes : **PixelSeal** (256 bits), **TrustMark** (variante Q, 100 bits), **StegaStamp** (100 bits).
- Deux façons de fabriquer les images : un **morceau** tatoué (TrustMark tatoue alors tout le morceau, de rapport largeur/hauteur au plus 2), ou l'image **entière** tatouée puis une partie remise à l'original (`full_watermark = true` dans les métadonnées du train).
- Évaluation sur une **grille 128×128** (cases de 4×4 pixels) : 1 si au moins la moitié de la case est tatouée, sinon 0.
- **Score** : moyenne, sur les images, du mIoU par image (IoU de la classe « tatoué » et de la classe « propre », moyennées ; 1 si une classe est absente des deux côtés). Mêmes fonctions que le serveur : `wmloc/utils.py` (`miou_image`, `score`).
- **Données** : train 510 images (170 par méthode) avec masque 512×512, `masks_128.npz` (la grille officielle), méthode, message, `full_watermark` ; test 1 020 images avec méthode et message.

## La soumission
- Un fichier `.pt` : `{"image_names": [noms sans .png], "masks": tenseur bool [1020, 128, 128]}`, chaque image de test une seule fois, 25 Mo au plus.
- Envoi : `submit(chemin)` de `submission_template.py` (régler `BASE_URL` et `API_KEY` ; probablement le même serveur que les autres tâches, `http://34.147.229.55`, à confirmer). **Sur le nœud de login** (internet). Clé : `os.environ["TEAM_API_KEY"]`, jamais dans git.

## Le kit officiel (`starter_kit.zip`)
- `wmloc/data.py` : `LocalisationDataset(root, split)`, éléments `image` (uint8 512×512×3), `image_name`, `method`, `message` (bits), et pour le train `mask`, `mask128`, `full_watermark`.
- `wmloc/watermarks.py` : `load_watermark("pixelseal" | "trustmark" | "stegastamp")`, les embedders exacts du jeu de données, avec `embed(image, bits)`, `logits(images)`, `decode(images)`, `features(images)`. PixelSeal et TrustMark exposent des **logits par position** (`per_position_logits`) avant la moyenne finale.
- `download_models.py` (une fois, **sur le nœud de login**), `submission_template.py` (baseline « tout à 0 » + score local).

## Installation (une fois, nœud de login, dans le dossier de la tâche)
```bash
unzip -o -q starter_kit.zip
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python download_models.py
python -c "from wmloc.data import unzip; unzip('train.zip', 'release'); unzip('test.zip', 'release')"
```
`requirements.txt` fixe torch 2.5.1 (CUDA 12.4) : compatible avec le pilote des GPU de JURECA.

## La baseline du kit : `watermark/baseline.py`
- **`--mode window`** (sans entraînement) : on connaît la méthode et le message de chaque image, donc on décode des **fenêtres glissantes** (128 et 256 px, pas de 64) avec le décodeur de la méthode, on note le taux de bits justes de chaque fenêtre, on moyenne sur les pixels, puis on seuille. Seuil réglé par méthode sur le train. Écrit `output/window.pt` et `output/window_maps.npz`.
- **`--mode unet`** : petit réseau de segmentation (image + passe-haut + méthode, et la carte du mode window si `--maps`) vers la grille 128×128, entraîné sur 80 % du train, mIoU sur les 20 % restants (par méthode). `--toutes` pour la version finale sur tout le train.
```bash
bash slurm/soumettre.sh 2 wm_window --venv watermark python watermark/baseline.py --mode window
bash slurm/soumettre.sh 2 wm_unet --venv watermark python watermark/baseline.py --mode unet \
    --maps $OFFICIEL/Hackathon_Watermark_Localisation_Task/output/window_maps.npz
```
Le score local (mIoU de validation) va dans le journal des scores. Test sans GPU : `python watermark/baseline.py --fake --mode unet --epochs 1`.

## Pistes
- **Données synthétiques illimitées** : les embedders sont fournis. Tatouer des images propres (régions propres du train, autres images) avec des messages aléatoires, puis recoller des morceaux de formes variées : autant de paires (image, masque) que voulu pour entraîner le réseau.
- **Logits par position** (`features(...)["per_position_logits"]`) pour PixelSeal et TrustMark : comparés au message connu, ils donnent une carte directe, plus fine que les fenêtres.
- **Rectangles** : pour les morceaux tatoués, chercher le rectangle dont le décodage colle le mieux au message.
- Régler un modèle ou un seuil **par méthode** : leurs signatures sont très différentes.
