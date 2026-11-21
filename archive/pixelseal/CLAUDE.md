# Tâche Pixel Seal (GPU 2)

> **Contexte du papier, pas la tâche réelle.** La tâche du hackathon est décrite dans `docs/TACHE_1_watermark.md`.

Le papier : *Pixel Seal*, arXiv 2512.16874. Code officiel : github.com/facebookresearch/videoseal (licence MIT), cloné dans `$DIR/repos/videoseal`.

## Le principe

- Un **embedder** (U-Net, 43,8 M de paramètres) produit un signal `w` qui contient un message de **256 bits**. L'image tatouée est `x_w = x + α·w`, avec α = `scaling_w` (0,2 par défaut), modulé par une carte JND qui cache le signal là où l'œil ne le voit pas.
- Un **extractor** (ConvNeXt-v2 Tiny, 33,4 M de paramètres) relit les 256 bits, même sur une image abîmée (JPEG, recadrage, luminosité, flou…).
- Le modèle travaille en 256×256, puis remet le tatouage à la taille d'origine de l'image.
- **La détection** : sur une image non tatouée, chaque bit est juste avec une probabilité de 1/2. La p-value binomiale donne la probabilité d'obtenir au moins autant de bits justes par hasard. Avec 256 bits, on peut se tromper sur 89 bits (65 % de bits justes) et rester sous p = 1e-6.
- **Le compromis central** : un tatouage plus fort (α plus grand) est plus robuste mais plus visible (PSNR plus bas).

## Le code

| Fichier | Rôle |
|---|---|
| `pixelseal/quickstart.py` | Tatoue N images, applique 6 attaques, relit le message. Calcule le PSNR, le taux de bits justes et −log10 p. Score = taux moyen de bits justes sur les attaques. |
| `pixelseal/run.sh` | Déposé dans la file du GPU 2 (`bash slurm/soumettre.sh 2 base bash pixelseal/run.sh`) : trois valeurs de `--scaling_w` (0,2, 0,3, 0,4) en parallèle sur le même GPU. |

L'API de videoseal :

```python
import videoseal
m = videoseal.load("pixelseal").cuda().eval()   # cartes : pixelseal, videoseal, chunkyseal (1024 bits)
m.blender.scaling_w = 0.3                        # force du tatouage
msg = m.get_random_msg(1)                        # 1 x 256
out = m.embed(img, msgs=msg)                     # img : B x 3 x H x W, float, valeurs dans [0, 1]
preds = m.detect(out["imgs_w"])["preds"]         # B x (1 + 256) : colonne 0 = détection
bits = preds[:, 1:] > 0
```

Évaluation officielle : `python -m videoseal.evals.full --checkpoint pixelseal --dataset <nom> ...`, depuis `$DIR/repos/videoseal`. Les jeux de données sont décrits dans `configs/datasets/*.yaml`.

## Les pièges

- **Les poids sont cherchés dans `./ckpts/` du dossier courant.** Lance toujours depuis la racine du dépôt, qui contient le lien `ckpts` vers `$DIR/ckpts`. Si les poids ne sont pas trouvés, la bibliothèque essaie de les télécharger, ce qui échoue sur le nœud GPU.
- `videoseal` s'installe avec `pip install -e . --no-deps`, après `requirements.txt`, pour éviter `decord` (déjà fait par `setup/install_pixelseal.sh`).
- Les attaques de `quickstart.py` tournent sur le processeur. Un GPU peu chargé est normal.
- `quickstart.py` tire un message aléatoire par image. Pour un test de détection avec p-value, utilise un message fixe.

## Les leviers

`scaling_w`, l'atténuation JND (`--attenuation` dans l'évaluation), le choix du modèle (`pixelseal`, `videoseal`, `chunkyseal`), un message fixe, et pour la vidéo `--videoseal_step_size`, `--videoseal_chunk_size` et `--time_pooling_depth`. Le réentraînement (`train.py`) est lourd : ne le lance que si un affinage ciblé rapporte beaucoup.

## À adapter dès qu'on connaît l'énoncé

- Remplace le score de `quickstart.py` par la métrique des tests, et ajoute leurs attaques à `ATTACKS`.
- Si la tâche se place du côté de l'attaquant (effacer ou falsifier un tatouage), le score devient la qualité d'image conservée face au taux de bits justes. Les points faibles se cherchent dans les attaques combinées et géométriques. Commence par mesurer avec `quickstart.py`.
