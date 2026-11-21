# Guide pour les agents — dépôt d'équipe du hackathon

Lis ce fichier en entier avant de toucher au code. Les détails sont dans :

- `docs/CLUSTER.md` : machines, Slurm, GPU, mode hors ligne ;
- `docs/STRATEGIE.md` : planning, règles de score, priorités ;
- `docs/ETAT.md` : l'état courant du travail. Lis-le au début, mets-le à jour à la fin ;
- **`docs/TACHE_1_watermark.md`, `docs/TACHE_2_model_equality.md`, `docs/TACHE_3_unlearning.md` : les vraies tâches, à lire avant de coder** ;
- `archive/` : le code écrit d'après les papiers avant de connaître les tâches. **Ne pas l'utiliser** (voir `archive/README.md`).

## Le contexte

Le hackathon (European AI Championship, Paris) va du samedi 21/11 à 14h30 au dimanche 22/11 à 12h ; ensuite, présentations des 6 meilleures équipes et jury. **Trois tâches, toutes obligatoires**, notées par le serveur des organisateurs (classement en direct, classement final sur une partie cachée des tests).

| Tâche | Ce qu'on rend | Score | Code du kit | Où ça tourne |
|---|---|---|---|---|
| 1. Localiser le tatouage (PixelSeal, TrustMark, StegaStamp) | une carte 128×128 par image | mIoU | `watermark/baseline.py` | file GPU 2 |
| 2. Model Equality : 33 endpoints d'API | CSV `endpoint,family,model` | 0,1 MCC famille + 0,9 MCC modèle | `met/analyse.py` | **nœud de login** (API, CPU) |
| 3. Unlearning : ViT-B/16, 3 500 images à oublier | `state_dict` (.pt) | forget × retain, 0 si précision < 70 % | `unlearning/unlearn.py` | file GPU 1 |

GPU 0 et 3 : libres pour les balayages et les entraînements longs (décision en équipe). La tâche 2 n'a pas besoin de GPU.

**Le dossier officiel `$OFFICIEL`** (`/p/scratch/training2666/huang25/Epi84`, créé par `hackathon_setup.sh`) contient les données et un `.venv` par tâche : `model_equality/`, `unlearning/`, `Hackathon_Watermark_Localisation_Task/`. Dans une file : `bash slurm/soumettre.sh <gpu> <nom> --venv unlearning|met|watermark <commande>`. Les soumissions et l'API passent par `34.147.229.55` en HTTP : **sur le nœud de login seulement** (les nœuds de calcul n'ont pas internet). **Clé d'API d'équipe : jamais dans git**, toujours `os.environ["TEAM_API_KEY"]`.

## Les règles absolues

1. **Tous les téléchargements se font sur le nœud de login**, celui auquel on se connecte avec `ssh jsc` : `pip install`, `git clone`, `wget`, `hf download`, jeux de données. **Les nœuds GPU n'ont pas internet** : un script qui y télécharge quelque chose plante.
2. **Rien sur les ordinateurs personnels.** Ni paquet, ni modèle, ni poids, ni données : l'ordinateur d'un membre sert uniquement de terminal SSH. Le code se modifie, se teste et se pousse (`git push`) depuis le clone du membre sur le nœud de login ; l'exécution a lieu sur le cluster.
3. **Pas de calcul lourd sur le nœud de login** : il est partagé avec tous les utilisateurs du centre. Tout ce qui utilise un GPU passe par Slurm (`docs/CLUSTER.md`).
4. **Un GPU par tâche.** Fixe toujours `CUDA_VISIBLE_DEVICES`. Dans le code, écris `"cuda"`, jamais `"cuda:2"` : le GPU choisi s'appelle toujours `cuda:0`.
5. **Chaque run écrit son score** avec `common.scores.log_score(tache, config, score, notes)`. Le journal est partagé (`$DIR/scores.csv`, via `SCORES_CSV` dans `slurm/env.sh`). Un résultat qui n'y est pas n'existe pas pour l'équipe.
6. **Ne commite jamais** de poids, de modèles, de données, `env/`, de jetons (Hugging Face, GitHub) ni de grosses sorties. Le `.gitignore` les exclut déjà : ne le contourne pas.
7. **Ne soumets rien au serveur (soumissions, et requêtes sur les endpoints `ep-` de la tâche 2, au budget limité) sans une décision de l'équipe, transmise par un humain** : le nombre de soumissions peut être limité.
8. **Petits tests libres, gros runs annoncés.** Un petit test, c'est une commande déposée dans la file du GPU de la tâche (`slurm/soumettre.sh`), de moins de 30 minutes, dont tu suis le journal : tu peux la lancer. Un run plus long ou de nuit, ou un gros téléchargement, s'annonce d'abord sur le canal d'équipe avec sa durée : tu demandes à l'humain de l'annoncer, et tu ne le lances qu'après sa confirmation. Le GPU 3 (joker) et les soumissions officielles se décident en équipe. N'annule jamais la commande d'un autre membre.
9. **Pas d'affirmation sans mesure.** Ne dis pas qu'un changement améliore le score tant qu'un run ne l'a pas montré dans `scores.csv`.

## L'arborescence

Voir `README.md`. L'essentiel : `watermark/baseline.py`, `met/analyse.py`, `unlearning/unlearn.py` (une baseline par tâche), `slurm/` (files d'attente des GPU), `common/scores.py` (journal partagé), `docs/` (tâches, cluster, état). `archive/` : ne pas utiliser.

Deux dossiers sur le cluster : `$DIR` = `/p/project1/training2666/team_Epi84` (files `queue/`, clones `membres/<id>/depot`, `scores.csv`) et `$OFFICIEL` = `/p/scratch/training2666/huang25/Epi84` (données et un `.venv` par tâche).

## Qui lance les jobs

- **Le titulaire du nœud**, un seul membre, porte le job sur son compte : `sbatch slurm/noeud.sh` garde le nœud et ouvre une file d'attente par GPU. Son seul rôle régulier : programmer la relève avec `bash slurm/relancer.sh` (un nouveau `noeud.sh` qui démarre à la fin du précédent, donc jamais deux nœuds payés). Il est seul à lancer des jobs Slurm. Chaque job réserve un nœud entier de 4 GPU : un deuxième job consomme un deuxième nœud et le double du budget.
- **Le relais de nuit**, à tour de rôle, prend le relais quand le titulaire dort. Si le job du nœud s'arrête, il relance `slurm/noeud.sh` sous son propre compte, car Slurm ne laisse annuler un job qu'à la personne qui l'a soumis.
- **Le responsable de chaque tâche** dépose ses commandes dans la file de son GPU (`bash slurm/soumettre.sh <gpu> <nom> <commande>`) et fait `git push` de ce qui marche. Chaque membre a son propre compte et sa propre clé SSH : personne n'entre dans le job d'un autre.
- **Toi, l'agent :** ne lance jamais `sbatch`, `salloc` ni `scancel` de ta propre initiative. Pour un petit test, tu peux utiliser `slurm/soumettre.sh` ; pour le reste, prépare le script et dis à l'humain qu'il est prêt. Ne modifie pas un script en attente ou en cours d'exécution dans une file : il est lu au démarrage, puis au fur et à mesure.

## Le cycle de travail

1. Lis `docs/ETAT.md` et le `CLAUDE.md` de la tâche.
2. Fais une modification, de préférence derrière une option en ligne de commande, pour garder la baseline reproductible.
3. Teste à petite échelle : `--fake` pour SCRUB, `--max_images 5` pour Pixel Seal, `--n 3 --max_tokens 20` pour MET.
4. Dépose la commande dans la file du GPU de la tâche : `bash slurm/soumettre.sh <gpu> <nom> <commande>`. Suis-la avec `bash slurm/file.sh` et `tail -f $DIR/queue/logs/<nom affiché>.log` (chemin affiché par `soumettre.sh`).
5. Vérifie que le score est bien écrit, puis compare avec `source slurm/env.sh && python -m common.scores`.
6. Si c'est mieux que le meilleur score : fais un commit avec un message de la forme `<tache>: <changement>, score <x> (<config>)`, puis `git push`. Si c'est moins bon : ajoute une ligne dans le journal de `docs/ETAT.md` (« essayé X, -0,02 »), pour que personne ne le relance.
7. Mets à jour `docs/ETAT.md` : meilleur score, configuration, commit, heure.
8. Juste après la baseline : remplace le score du kit par la métrique des tests. Objectif : les trois tâches notées avant 16h, avant toute amélioration.
9. Soumission officielle : uniquement avec l'accord de l'équipe. Marque le commit soumis avec `git tag <tache>-soumis-<hh>h`.
10. Dimanche 10h, gel du code : plus d'expérience. Les runs finaux repartent du meilleur commit marqué.

## Ce qui a été testé

- **Sur JURECA :** GPU (4 × A100), files d'attente (`noeud.sh`, `soumettre.sh`, `file.sh`), partage du dossier, `relancer.sh`, le setup officiel.
- **Sur données factices seulement (`--fake`) :** les trois baselines. Elles n'ont pas encore tourné sur les vraies données : au premier échec, lis le journal avant de réécrire.
- `met/analyse.py` a été vérifié sur des réponses synthétiques imitant les 9 endpoints de référence.

## Le style

- Le code, les commentaires et les messages sont en français, comme le reste du dépôt.
- Les scripts se lancent depuis la racine du dépôt (`python met/analyse.py …`) ; ils ajoutent eux-mêmes la racine à `sys.path`.
- Chaque nouvel hyperparamètre devient une option `argparse` et apparaît dans le `config` passé à `log_score`.
- Une graine explicite pour tout ce qui est aléatoire.
