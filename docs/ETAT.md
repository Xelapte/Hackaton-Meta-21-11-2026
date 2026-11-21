# État du travail

Ce fichier est le relais entre les membres et les agents. Lis-le en arrivant. Mets-le à jour en partant : ce qui a été fait, ce qui tourne, ce qui reste à faire, avec l'heure. Garde-le court : seuls les meilleurs scores et les décisions y figurent ; le détail est dans le journal partagé `$DIR/scores.csv`.

## Ce qu'on sait des tests

| Tâche | Format rendu | Métrique | Soumissions | Détails |
|---|---|---|---|---|
| 1 Tatouage | carte 128×128 par image | mIoU (plus haut = mieux) | à préciser (README) | docs/TACHE_1_watermark.md |
| 2 Model Equality | CSV endpoint,family,model | 0,1 MCC famille + 0,9 MCC modèle | 1 / 30 min ; budget 100 réponses/endpoint | docs/TACHE_2_model_equality.md |
| 3 Unlearning | state_dict .pt | forget × retain, 0 si précision < 70 % | à préciser | docs/TACHE_3_unlearning.md |

## Budget de la tâche 2 (à tenir à jour)

| Endpoints | Réponses utilisées / 100 | Plan |
|---|---|---|
| ep-01 … ep-33 | | |

## Les meilleurs scores

| Tâche | Score | Configuration | Commit | Heure |
|---|---|---|---|---|
| Tatouage (mIoU local) | 0,4705 (hors pli, 5 plis) | `--mode unet --feats cartes_feats.npz --folds 5 --aug --epochs 40` (après `--mode cartes`) | 9d78d06 | 17h28 |
| Tatouage, sans entraînement | 0,4339 (train) | `--mode direct` | 9d78d06 | 17h20 |
| MET | | | | |
| SCRUB | | | | |

## Ce qui tourne en ce moment

Pour l'instant précis : `bash slurm/file.sh`. Ce tableau sert aux longs runs et à la nuit : chacun y inscrit les siens en les annonçant. Les petits tests (moins de 30 minutes, dans la file de son propre GPU) sont libres.

| GPU | Commande (nom dans la file) | Quoi | Déposée par | Fin prévue |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## Qui tient le nœud

- Titulaire du nœud (son compte porte `slurm/noeud.sh`) : 
- Relais de nuit (à tour de rôle) : 
- Fin du job en cours (`--time`) : 

## À faire

- [ ] Remplir `DIR` dans `slurm/env.sh`, `bash setup/install_env.sh`, puis faire passer `slurm/test.sh`.
- [ ] `setup/partage.sh <dossier_projet> <identifiants...>`, un clone par membre dans `$DIR/membres/`, `umask 002` pour tous.
- [ ] `sbatch slurm/noeud.sh` : `bash slurm/file.sh` montre 4 files actives.
- [ ] Baseline Pixel Seal adaptée à la métrique des tests.
- [ ] Baseline MET adaptée à la métrique des tests.
- [ ] Baseline SCRUB adaptée à la métrique des tests.

## Le journal des décisions

- _hh:mm_ — décision prise, et pourquoi.
- 17h20 — Tatouage : les logits par position corrélés au message connu (PixelSeal 512 px, TrustMark 224 px) battent les fenêtres glissantes (TrustMark 0,29 → 0,47). StegaStamp : « re-tatouer et corréler » (R' = embed(I, msg) − I, corrélation locale, bloc 16) 0,26 → 0,36. Le canal de détection de PixelSeal (canal 0) est faible (0,33) : essayé, abandonné. Le re-tatouage de PixelSeal et TrustMark ne marche pas (~0,27).
- 17h30 — Tatouage, diagnostic : la part tatouée est bimodale (1/3 < 10 %, 1/3 > 90 %). L'oracle « classe majoritaire » vaut 0,455 ; on est à 0,47 : on ne voit presque pas la petite zone, c'est là qu'est le reste du score. Piste en cours : données synthétiques (`watermark/synthetique.py` : trous, objets transplantés, rectangles) pour entraîner le U-Net.
