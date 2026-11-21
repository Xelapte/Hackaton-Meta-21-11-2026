# Tâche 3 : Unlearning (faire oublier 3 500 images à un ViT)

Dossier officiel : `$OFFICIEL/unlearning/` (venv : `--venv unlearning`). Fichiers fournis : `model.pt`, `forget_images.npz`, `task_template.py`, `submission.py`.

## Le problème
- **Modèle** : ViT-B/16 (`timm.create_model("vit_base_patch16_224", num_classes=100, init_values=1e-4)`), entraîné sur un jeu de données à 100 classes.
- **À oublier** : 3 500 images (`forget_images.npz`, clé `"images"`, uint8 `[3500, 224, 224, 3]`), **sans étiquettes**. **Aucune autre donnée n'est fournie**, donc pas de retain set.
- **Prétraitement** (`preprocess` du template) : `/255`, puis `(x - 0,5) / 0,5`. Pas de redimensionnement, ne pas normaliser deux fois.
- **Soumission** : un `state_dict` de tenseurs float32 (sans NaN ni Inf), chargeable en `strict=True` dans `build_model()`, 450 Mo au plus, envoyé avec `submission.py` (`.pt`).

## Le score
`score = forget_score × retain_score`, et **0 si la précision sur les données tenues à part passe sous 70 %**.
- **forget_score** = score de distribution des pertes × score de prédiction, mesurés sur les images à oublier par rapport au modèle **gold** (réentraîné sans elles, moyenne de plusieurs modèles) :
  - pertes : `1 − clip(d_sub / d_orig, 0, 1)`, avec `d = |Δ perte moyenne| + |Δ écart-type|` par rapport à gold ;
  - prédictions : `1 − clip(KL(gold‖sub) / KL(gold‖orig), 0, 1)`, image par image, softmax à T = 2.
- **retain_score** = précision × similarité KL, sur un jeu de contrôle :
  - précision : `clip(acc_sub / acc_orig, 0, 1)` ;
  - similarité : `1 / (1 + KL moyen(orig‖sub))`, donc les sorties doivent rester proches de l'original.

## Ce que ça implique
- Gold n'a jamais vu ces images : sur elles, il se comporte comme le modèle original sur des **images inconnues** (souvent la bonne classe, mais moins confiant, perte plus haute et plus dispersée). La cible n'est **pas** de tout casser : sur-oublier fait chuter le score de prédiction.
- Toute modification qui déborde sur les autres images coûte au retain_score : changements petits et ciblés.
- On ne peut pas calculer le score localement (gold et le jeu de contrôle sont sur le serveur) : **le serveur est le seul juge**. Notez chaque soumission (réglages, score) dans `docs/ETAT.md`.

## La baseline du kit : `unlearning/unlearn.py`
Distillation vers les prédictions **adoucies** du modèle original sur les images à oublier (softmax à température `--T`), avec une ancre L2 vers les poids d'origine.
```bash
bash slurm/soumettre.sh 1 ul_T2 --venv unlearning python unlearning/unlearn.py \
    --T 2.0 --steps 300 --lr 1e-5 --params all --out $OFFICIEL/unlearning/output/T2.pt
```
Puis, sur le nœud de login (internet), avec le venv activé : modifiez `SUBMISSION_FILE` (ou équivalent) dans `submission.py` et lancez `python submission.py`. Gardez la clé d'API hors de git : `os.environ["TEAM_API_KEY"]`.

Leviers : `--T` (force de l'oubli : 1 = rien ne change), `--steps`, `--lr`, `--params all|last|norm|head` (quelles couches bougent), `--anchor`. Le script affiche la perte moyenne, l'écart-type, la confiance et l'accord avec l'original avant et après.

## Pistes
- Balayer `T` (1,5 → 4) et `--params` : c'est le principal compromis oubli/rétention.
- Faire du « contrôle » avec des images proches de la distribution (augmentations fortes des images à oublier, mélanges) pour garder les sorties stables ailleurs.
- SCRUB (fiche 3, `scrub/scrub.py`) demande un retain set : à adapter avec des données de substitution.
- Garder une marge sur la précision : sous 70 %, le score tombe à 0.
