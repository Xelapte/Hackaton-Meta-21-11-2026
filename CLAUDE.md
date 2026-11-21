# Guide pour les agents — dépôt d'équipe du hackathon

Lis ce fichier en entier avant de toucher au code. Les détails sont dans :

- `docs/CLUSTER.md` : machines, Slurm, GPU, mode hors ligne ;
- `docs/STRATEGIE.md` : planning, règles de score, priorités ;
- `docs/ETAT.md` : l'état courant du travail. Lis-le au début, mets-le à jour à la fin ;
- `pixelseal/CLAUDE.md`, `met/CLAUDE.md` et `scrub/CLAUDE.md` : une tâche chacun.

## Le contexte en trois lignes

Le hackathon dure 24 h, du samedi 21/11 à 12h au dimanche 22/11 à 12h, puis vient le pitch devant le jury jusqu'à 15h. Il y a **trois tâches, toutes obligatoires**, et chacune est notée par des tests fournis par le jury. Le score de chaque tâche compte dans le classement.

| Tâche | Papier | Dossier | GPU |
|---|---|---|---|
| Tatouage d'images (watermarking) | Pixel Seal, arXiv 2512.16874 | `pixelseal/` | 2 |
| Audit d'API de modèles de langage | Model Equality Testing (MET), arXiv 2410.20247 | `met/` | 0 |
| Désapprentissage (machine unlearning) | SCRUB, arXiv 2302.09880 | `scrub/` | 1 |

Le GPU 3 est un « joker » : l'équipe le réattribue selon les scores, puis le réserve à la démo.

Toutes les équipes ont les mêmes papiers. Les papiers donnent la baseline ; les points se gagnent en optimisant **la métrique des tests du jury**, qui n'est pas forcément celle du papier.

## Les règles absolues

1. **Tous les téléchargements se font sur le nœud de login**, celui auquel on se connecte avec `ssh jsc` : `pip install`, `git clone`, `wget`, `hf download`, jeux de données. **Les nœuds GPU n'ont pas internet** : un script qui y télécharge quelque chose plante.
2. **Rien sur les ordinateurs personnels.** Ni paquet, ni modèle, ni poids, ni données. Si tu tournes sur l'ordinateur d'un membre, tu modifies le code et tu fais `git push`, c'est tout : l'exécution a lieu sur le cluster.
3. **Pas de calcul lourd sur le nœud de login** : il est partagé avec tous les utilisateurs du centre. Tout ce qui utilise un GPU passe par Slurm (`docs/CLUSTER.md`).
4. **Un GPU par tâche.** Fixe toujours `CUDA_VISIBLE_DEVICES`. Dans le code, écris `"cuda"`, jamais `"cuda:2"` : le GPU choisi s'appelle toujours `cuda:0`.
5. **Chaque run écrit son score** avec `common.scores.log_score(tache, config, score, notes)`. Un résultat qui n'est pas dans `results/scores.csv` n'existe pas pour l'équipe.
6. **Ne commite jamais** de poids, de modèles, de données, `env/`, de jetons (Hugging Face, GitHub) ni de grosses sorties. Le `.gitignore` les exclut déjà : ne le contourne pas.
7. **Ne soumets rien aux tests officiels sans l'accord d'un humain** : le nombre de soumissions peut être limité.
8. **Pas d'affirmation sans mesure.** Ne dis pas qu'un changement améliore le score tant qu'un run ne l'a pas montré dans `scores.csv`.

## L'arborescence

```
CLAUDE.md                  ce fichier
docs/                      CLUSTER.md, STRATEGIE.md, ETAT.md
common/scores.py           log_score(...) ; python -m common.scores [--bas met]
common/dashboard.py        python -m common.dashboard -> results/dashboard.html
common/smoke_test.py       test de fumée GPU (lancé par slurm/test.sh)
setup/                     install_env.sh, download_all.sh, install_pixelseal.sh (nœud de login)
slurm/env.sh               variables communes : DIR, MODEL_PATH, IMAGES, HF_HUB_OFFLINE=1
slurm/test.sh              job de test de 10 minutes
slurm/vague.sh             un processus par GPU : lance met/, scrub/ et pixelseal/run.sh
slurm/shell.sh             terminal interactif limité à un GPU, dans une réservation
pixelseal/ met/ scrub/     code de chaque tâche ; run.sh = ce que la vague exécute
results/scores.csv         journal des scores (versionné)
logs/                      sorties des jobs (non versionnées)
```

Les gros fichiers vivent hors du dépôt, dans le dossier projet du cluster (`$DIR`) : `env/`, `models/`, `data/`, `ckpts/`, `repos/` et `hf_cache/`. Le dépôt contient seulement un lien symbolique `ckpts`, qui pointe vers `$DIR/ckpts`.

## Le cycle de travail

1. Lis `docs/ETAT.md` et le `CLAUDE.md` de la tâche.
2. Fais une modification, de préférence derrière une option en ligne de commande, pour garder la baseline reproductible.
3. Teste à petite échelle : `--fake` pour SCRUB, `--max_images 5` pour Pixel Seal, `--n 3 --max_tokens 20` pour MET.
4. Lance sur le GPU de la tâche, avec `bash slurm/shell.sh <jobid> <gpu>` ou en modifiant `<tache>/run.sh` puis `sbatch slurm/vague.sh`.
5. Vérifie que le score est bien écrit, puis compare avec `python -m common.scores`.
6. Fais un commit avec un message qui donne la configuration et le score obtenu. Mets à jour `docs/ETAT.md`.

## Ce qui a été testé

- **Testé, sans GPU :** `common/scores.py` (écritures simultanées), `common/dashboard.py`, `met/mmd_test.py` et `met/compare.py` (taux de faux positifs de 5 % sous H0 sur données synthétiques), et toute la chaîne SCRUB sur données factices (`--fake`).
- **Vérifié à blanc**, sans GPU, avec un faux `python` : l'enchaînement des `run.sh` et les gardes des scripts Slurm.
- **Jamais exécuté** (ni GPU ni poids disponibles pendant la préparation) : `pixelseal/quickstart.py`, `met/generate_vllm.py`, `met/generate_hf.py`, `slurm/shell.sh`, et les jobs Slurm sur la vraie machine. Au premier échec, lis l'erreur avant de réécrire quoi que ce soit.

## Le style

- Le code, les commentaires et les messages sont en français, comme le reste du dépôt.
- Les scripts se lancent depuis la racine du dépôt (`python met/compare.py …`) ; ils ajoutent eux-mêmes la racine à `sys.path`.
- Chaque nouvel hyperparamètre devient une option `argparse` et apparaît dans le `config` passé à `log_score`.
- Une graine explicite pour tout ce qui est aléatoire.
