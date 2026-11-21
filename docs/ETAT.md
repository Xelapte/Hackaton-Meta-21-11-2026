# État du travail

Ce fichier est le relais entre les membres et les agents. Lis-le en arrivant. Mets-le à jour en partant : ce qui a été fait, ce qui tourne, ce qui reste à faire, avec l'heure. Garde-le court : seuls les meilleurs scores et les décisions y figurent ; le détail est dans le journal partagé `$DIR/scores.csv`.

## Ce qu'on sait des tests

| Tâche | Format rendu | Métrique | Soumissions | Détails |
|---|---|---|---|---|
| 1 Tatouage | carte 128×128 par image | mIoU (plus haut = mieux) | public 30 % / final 70 % ; meilleur gardé | docs/TACHE_1_watermark.md |
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
| MET (MCC modèle, ref-) | 0,231 | `met/score_local.py` (LLR SmolLM2-360M base − fp8, seuil z 2) : fp8 repéré, LoRA non séparées | ec720e8 | 20h37 |
| SCRUB | | | | |

## Ce qui tourne en ce moment

Pour l'instant précis : `bash slurm/file.sh`. Ce tableau sert aux longs runs et à la nuit : chacun y inscrit les siens en les annonçant. Les petits tests (moins de 30 minutes, dans la file de son propre GPU) sont libres.

| GPU | Commande (nom dans la file) | Quoi | Déposée par | Fin prévue |
|---|---|---|---|---|
| 0 | wm_fin_ss0, wm_fin_ps0 | Tâche 1 : recherche fine des rectangles (train) | huang25 | ~21h25 |
| 1 | | | | |
| 2 | wm_fin_ss1, wm_fin_ps1 | Tâche 1 : recherche fine (train) | huang25 | ~21h25 |
| 3 | wm_fin_ss2, wm_fin_ss3 | Tâche 1 : recherche fine (train) ; puis le test (~2 h 25) si le gain est robuste | huang25 | ~21h25 (test ~23h55) |

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
- 17h46 — **Soumission tatouage n° 1136** (demandée par huang25) : `output/unet_feats_cv5.pt`, tag `watermark-soumis-17h` (commit 9d78d06). Public **0,4804** (avant : 0,3321). Hors pli local : 0,4705.
- 18h15 — Tâche 1 : protocole anti-sur-apprentissage (plis gelés `watermark/splits.json`, journal `docs/RECHERCHE_WATERMARK.md`). GPU 3 attribué au watermark par huang25.
- 20h04 — **Soumission tatouage n° 1412** (demandée par l'utilisateur) : H3b, rectangle + repli carte (`regle.py --predire --repli carte`, `output/regle_h3b_carte.pt`). Public **0,5042** (avant : 0,4804). Protocole (plis gelés) : 0,5065 ± 0,0092. Tag `watermark-soumis-20h`.
- 20h06 — Tâche 1 : recherche fine (H8b) lancée sur les GPU 0, 2 et 3 (accord de huang25), morceaux de <= 40 min pour laisser passer l'unlearning sur le GPU 0.
- 20h37 — Tâche 2 : le modèle de base des ref- est **SmolLM2-360M-Instruct** (log-probas nettement meilleures que 135M et 1.7B). Le LLR base − fp8 simulé (`met/score_local.py`) isole ref-02 (fp8, seule négative), les 5 S restent positives malgré les déguisements. MCC modèle 0,231 sur les ref- : les LoRA passent pour S. Toutes les ref- échantillonnent (aucune répétition identique).
- 21h00 — Tâche 2, **essayé sans succès** (`met/logprobs_local.py` + `met/analyse_logprobs.py`) : pour séparer les LoRA, statistiques token par token sous le 360M local (température ajustée, vraisemblance, part d'argmax, PIT/KS, tokens loin du début, tokens à faible entropie). La LoRA 1,5 se voit, la LoRA 0,7 se confond avec persona/prompt système, la LoRA 0,3 est indiscernable de S. Seul indice : sur le prompt long et neutre (forêt), les déguisements ne bougent presque pas la log-proba (−0,78 à −0,82 vs −0,88) alors que LoRA 0,7 → −0,95 ; à vérifier avec plus de prompts neutres longs (les données ref de l'équipe n'ont que 5 prompts distincts). Environnement : transformers dans `$DIR/membres/bouaboud1/pylibs_met` (PYTHONPATH) + venv unlearning ; modèles dans `$DIR/models/met_local/`.
- 20h37–20h47 — Le job du nœud s'est arrêté (fin de `--time`) ; relais 15900958 démarré à 20h47. Les runs `wm_fin_ss0/ss1/ss2` de huang25 ont été coupés en cours (restés dans `en_cours/`).
- 21h25 — **Quota disque du scratch atteint** (quota de données par utilisateur, huang25) : plus aucune écriture possible dans `$OFFICIEL`. Libéré ~21 Go en supprimant les données synthétiques rejetées (`output/synth/`, images de `output/synth_v2/`, `test_*`). Attention aux sorties volumineuses : vérifier `jutil user dataquota` et `du -sh $OFFICIEL/*/output` avant un gros run.
- 22h55 — **Tâche 2 : collecte de la phase A lancée sur les 33 ep-** (décision de huang25) : `met/prompts_ep_phaseA.csv`, 10 prompts, 30 réponses/endpoint (990), `collecte.py --plafond_serveur 30`. Sortie : `$OFFICIEL/model_equality/output/collecte_ep.jsonl`. **Ne pas interroger l'API en parallèle** (limite de 4 requêtes par équipe) jusqu'à la fin (~0h45). Budget après la phase A : 70/100 par endpoint.
