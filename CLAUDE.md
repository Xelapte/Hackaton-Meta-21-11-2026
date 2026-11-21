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
5. **Chaque run écrit son score** avec `common.scores.log_score(tache, config, score, notes)`. Le journal est partagé (`$DIR/scores.csv`, via `SCORES_CSV` dans `slurm/env.sh`). Un résultat qui n'y est pas n'existe pas pour l'équipe.
6. **Ne commite jamais** de poids, de modèles, de données, `env/`, de jetons (Hugging Face, GitHub) ni de grosses sorties. Le `.gitignore` les exclut déjà : ne le contourne pas.
7. **Ne soumets rien aux tests officiels sans l'accord d'un humain** : le nombre de soumissions peut être limité.
8. **Petits tests libres, gros runs annoncés.** Un petit test, c'est une commande déposée dans la file du GPU de la tâche (`slurm/soumettre.sh`), de moins de 30 minutes, dont tu suis le journal : tu peux la lancer. Un run plus long ou de nuit, ou un gros téléchargement, s'annonce d'abord sur le canal d'équipe avec sa durée : tu demandes à l'humain de l'annoncer, et tu ne le lances qu'après sa confirmation. Le GPU 3 (joker) et les soumissions officielles se décident en équipe. N'annule jamais la commande d'un autre membre.
9. **Pas d'affirmation sans mesure.** Ne dis pas qu'un changement améliore le score tant qu'un run ne l'a pas montré dans `scores.csv`.

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
slurm/noeud.sh             job du titulaire : garde le nœud, une file d'attente par GPU (worker.sh)
slurm/soumettre.sh         déposer une commande dans la file d'un GPU
slurm/file.sh, annuler.sh  état des files ; arrêter la commande en cours
slurm/vague.sh, shell.sh   secours (titulaire seulement)
setup/partage.sh           crée les files, droits du groupe (une fois)
pixelseal/ met/ scrub/     code de chaque tâche ; run.sh = la baseline complète
logs/                      sorties des jobs (non versionnées)
```

Les gros fichiers vivent hors du dépôt, dans le dossier projet du cluster (`$DIR`) : `env/`, `models/`, `data/`, `ckpts/`, `repos/` et `hf_cache/`. Le dépôt contient seulement un lien symbolique `ckpts`, qui pointe vers `$DIR/ckpts`. Chaque membre a son clone dans `$DIR/membres/<identifiant>/`. Les files d'attente sont dans `$DIR/queue/`, le journal des scores dans `$DIR/scores.csv`.

## Qui lance les jobs

- **Le titulaire du nœud**, un seul membre, porte le job sur son compte : `sbatch slurm/noeud.sh` garde le nœud et ouvre une file d'attente par GPU. Son seul rôle régulier : relancer ce job avant la fin de sa durée. Il est seul à lancer des jobs Slurm. Chaque job réserve un nœud entier de 4 GPU : un deuxième job consomme un deuxième nœud et le double du budget.
- **Le relais de nuit**, à tour de rôle, prend le relais quand le titulaire dort. Si le job du nœud s'arrête, il relance `slurm/noeud.sh` sous son propre compte, car Slurm ne laisse annuler un job qu'à la personne qui l'a soumis.
- **Le responsable de chaque tâche** dépose ses commandes dans la file de son GPU (`bash slurm/soumettre.sh <gpu> <nom> <commande>`) et fait `git push` de ce qui marche. Chaque membre a son propre compte et sa propre clé SSH : personne n'entre dans le job d'un autre.
- **Toi, l'agent :** ne lance jamais `sbatch`, `salloc` ni `scancel` de ta propre initiative. Pour un petit test, tu peux utiliser `slurm/soumettre.sh` ; pour le reste, prépare le script et dis à l'humain qu'il est prêt. Ne modifie pas un script en attente ou en cours d'exécution dans une file : il est lu au démarrage, puis au fur et à mesure.

## Le cycle de travail

1. Lis `docs/ETAT.md` et le `CLAUDE.md` de la tâche.
2. Fais une modification, de préférence derrière une option en ligne de commande, pour garder la baseline reproductible.
3. Teste à petite échelle : `--fake` pour SCRUB, `--max_images 5` pour Pixel Seal, `--n 3 --max_tokens 20` pour MET.
4. Dépose la commande dans la file du GPU de la tâche : `bash slurm/soumettre.sh <gpu> <nom> <commande>`. Suis-la avec `bash slurm/file.sh` et `tail -f $DIR/queue/logs/<nom>.log`.
5. Vérifie que le score est bien écrit, puis compare avec `source slurm/env.sh && python -m common.scores`.
6. Si c'est mieux que le meilleur score : fais un commit avec un message de la forme `<tache>: <changement>, score <x> (<config>)`, puis `git push`. Si c'est moins bon : ajoute une ligne dans le journal de `docs/ETAT.md` (« essayé X, -0,02 »), pour que personne ne le relance.
7. Mets à jour `docs/ETAT.md` : meilleur score, configuration, commit, heure.
8. Juste après la baseline : remplace le score du kit par la métrique des tests. Objectif : les trois tâches notées avant 16h, avant toute amélioration.
9. Soumission officielle : uniquement avec l'accord de l'équipe. Marque le commit soumis avec `git tag <tache>-soumis-<hh>h`.
10. Dimanche 10h, gel du code : plus d'expérience. Les runs finaux repartent du meilleur commit marqué.

## Ce qui a été testé

- **Testé, sans GPU :** `common/scores.py` (écritures simultanées), `common/dashboard.py`, `met/mmd_test.py` et `met/compare.py` (taux de faux positifs de 5 % sous H0 sur données synthétiques), et toute la chaîne SCRUB sur données factices (`--fake`).
- **Vérifié à blanc**, sans GPU, avec un faux `python` : l'enchaînement des `run.sh` et les gardes des scripts Slurm.
- **Jamais exécuté** (ni GPU ni poids disponibles pendant la préparation) : `pixelseal/quickstart.py`, `met/generate_vllm.py`, `met/generate_hf.py`, `slurm/shell.sh`, et les jobs Slurm sur la vraie machine. Les files d'attente (`noeud.sh`, `worker.sh`, `soumettre.sh`, `file.sh`, `annuler.sh`) ont été testées sans GPU ni Slurm : dépôt, ordre, annulation, arrêt, reprise. Au premier échec, lis l'erreur avant de réécrire quoi que ce soit.

## Le style

- Le code, les commentaires et les messages sont en français, comme le reste du dépôt.
- Les scripts se lancent depuis la racine du dépôt (`python met/compare.py …`) ; ils ajoutent eux-mêmes la racine à `sys.path`.
- Chaque nouvel hyperparamètre devient une option `argparse` et apparaît dans le `config` passé à `log_score`.
- Une graine explicite pour tout ce qui est aléatoire.
