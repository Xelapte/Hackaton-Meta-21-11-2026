# Le cluster : machines, Slurm, GPU

Les GPU sont au centre de calcul de Jülich (JSC). On y accède par JuDoor. Les noms exacts (machine, projet, partition, réservation) sont donnés par les organisateurs : ils remplacent les valeurs entre chevrons.

## Les trois machines

| Machine | Internet | GPU | Rôle |
|---|---|---|---|
| Ordinateur d'un membre | oui | non | Terminal SSH seulement. **Rien d'installé, rien de téléchargé.** |
| Nœud de login (`ssh jsc`) | oui | non | **Tous** les téléchargements et installations, `git`, dépôt des commandes dans les files. Pas de calcul lourd. |
| Nœud GPU | **non** | 4 × 40 Go | Exécution des commandes des files, dans le job du titulaire. On ne s'y connecte pas. |

Le nœud de login et le nœud GPU voient le même dossier projet, `$DIR`, placé sur un espace partagé comme `$PROJECT` ou `$SCRATCH` (à confirmer avec les organisateurs). Ce qui est téléchargé sur le nœud de login est donc lu tel quel sur le nœud GPU.

## La connexion

Chaque membre a **sa propre** clé SSH pour JSC, créée sur son ordinateur : `~/.ssh/id_ed25519-JSC` ; on ne partage jamais une clé. Sa partie publique est déposée dans JuDoor, avec une clause `from="…"` qui limite les adresses IP autorisées. L'alias `jsc` est défini dans `~/.ssh/config` : `ssh jsc`, `scp fichier jsc:`.

Si la connexion est refusée, c'est presque toujours parce que l'adresse IP a changé, par exemple sur le Wi-Fi du lieu. Il faut alors ajouter la plage d'adresses dans JuDoor.

L'accès à GitHub depuis le nœud de login se fait avec une **autre** clé, créée sur le nœud de login et déposée sur GitHub.

## Le dossier projet ($DIR)

```
$DIR/env/        ancien environnement Python (vLLM) : les tâches utilisent les venvs officiels
$DIR/repos/      dépôts officiels : videoseal, model-equality-testing, SCRUB
$DIR/ckpts/      poids de Pixel Seal (pixelseal_checkpoint.pth, videoseal_y_256b_img.pth)
$DIR/models/     modèle de langage pour MET (hf download)
$DIR/data/       CIFAR-10 et CIFAR-100
$DIR/hf_cache/   cache Hugging Face (HF_HOME)
$DIR/membres/    un clone du dépôt par membre
$DIR/queue/      files d'attente des GPU (gpu0 … gpu3) et leurs journaux (logs/)
$DIR/scores.csv  journal des scores de toute l'équipe
```

## L'installation (faite le 21/11)

- Setup officiel : `source ~/hackathon_setup.sh` (titulaire), `source teammate.sh` (chaque membre), avec `TEAM_LEAD="huang25"`, `TEAM_FOLDER="Epi84"`. Données et venvs dans `$OFFICIEL`.
- Tâche 1 : starter kit et venv construits à la main (voir `docs/TACHE_1_watermark.md`).
- Dossier d'équipe `$DIR` et files : `setup/partage.sh`, puis `sbatch slurm/noeud.sh` et `bash slurm/relancer.sh` (titulaire).

## Lancer sur les GPU : les files d'attente

Chaque membre a **son propre compte et sa propre clé SSH** ; on ne partage jamais une clé. Slurm ne laisse entrer dans un job que la personne qui l'a lancé. Les autres membres ne peuvent donc pas ouvrir de terminal sur le nœud du titulaire. D'où ce fonctionnement :

```
ordinateur --ssh jsc--> nœud de login --dépose--> $DIR/queue/gpuN/attente/ --exécute--> nœud de calcul, GPU N
                         (votre clone)              (dossier partagé)           (job slurm/noeud.sh du titulaire)
```

**Une fois.**

- Le titulaire lance `bash setup/partage.sh <dossier_projet> <id1> <id2> ...` avec les identifiants JuDoor de tous les membres : il crée les files, ferme le dossier au groupe Unix du projet (commun à toutes les équipes) et l'ouvre aux seuls membres par ACL. Un membre ajouté plus tard : chaque membre déjà présent lance `bash setup/ajouter_membre.sh <dossier_projet> <identifiant>` pour ses propres fichiers (Linux ne laisse changer les droits qu'au propriétaire).
- Chaque membre ajoute `umask 002` à son `~/.bashrc` et clone le dépôt dans `$DIR/membres/<identifiant>/`.
- Le titulaire lance `sbatch slurm/noeud.sh` : un seul job, qui garde le nœud et exécute une file par GPU. Il programme la relève dès le début avec `bash slurm/relancer.sh` (`sbatch --dependency=afterany`) : le job suivant démarre à la fin du précédent, jamais deux nœuds payés. Une commande coupée par la fin du job est classée « interrompue » ; les commandes en attente ne sont pas perdues.

**À chaque run**, depuis la racine de ton clone, sur le nœud de login :

```bash
bash slurm/soumettre.sh <gpu> <nom_court> <commande...>   # ex. : 1 ul_T2 --venv unlearning python unlearning/unlearn.py
bash slurm/file.sh                                        # état des 4 files
tail -f $DIR/queue/logs/<nom affiché>.log                  # suivre le run (chemin affiché par soumettre.sh)
bash slurm/annuler.sh <gpu>                               # arrêter la commande en cours
touch $DIR/queue/STOP                                     # titulaire : fermer les files proprement
```

Le fonctionnement des files :

- Une file exécute une commande à la fois, dans l'ordre de dépôt, avec `slurm/env.sh` chargé et `CUDA_VISIBLE_DEVICES` fixé.
- Pour plusieurs runs en parallèle sur un GPU, mets-les dans un même script avec `&` puis `wait`.
- La commande lit les fichiers au démarrage, pas au dépôt. Ne modifie pas un script en attente ou en cours.
- Le journal des scores est partagé : `SCORES_CSV=$DIR/scores.csv`, fixé dans `slurm/env.sh`.

**La règle.** Les petits tests sont libres : dans la file de ton propre GPU, en moins de 30 minutes, en suivant le journal. Un run long ou de nuit, ou un gros téléchargement, s'annonce sur le canal d'équipe avec sa durée, et part si personne n'objecte dans les 5 minutes. Le GPU 3 (joker) et les soumissions officielles se décident aux points d'équipe. On n'annule jamais la commande d'un autre sans lui. Seul le titulaire du nœud lance des jobs Slurm ; la nuit, un relais à tour de rôle relance `noeud.sh` si le job s'arrête.

Les files ont été testées sans GPU : dépôt, ordre, annulation, arrêt, reprise après interruption. Elles restent à valider sur la vraie machine. Les solutions de secours :

- `slurm/shell.sh`, pour le titulaire seulement, ouvre un terminal dans son propre job.

La machine est JURECA-DC : compte `training2666`, partition `dc-gpu`, réservation `cispahack` (jusqu'au 22/11 14h). Ces valeurs sont déjà dans les lignes `#SBATCH`.

## Les règles du GPU partagé

- Fixe toujours `CUDA_VISIBLE_DEVICES`. Dans le code, utilise `"cuda"` : le GPU choisi s'appelle `cuda:0`.
- `nvidia-smi` ignore `CUDA_VISIBLE_DEVICES` et montre les quatre GPU. C'est utile pour voir qui utilise quoi, pas pour vérifier ce que voit ton programme.
- vLLM réserve 90 % de la mémoire du GPU par défaut (`--gpu_mem 0.90`). Pour partager un GPU avec un autre processus, baisse cette valeur.
- Les cœurs du processeur et la mémoire vive sont partagés aussi : `nproc` donne le nombre de cœurs. Limite le nombre de processus de chargement de données.
- `slurm/env.sh` fixe `HF_HUB_OFFLINE=1`. Si un script essaie quand même de télécharger quelque chose, ne retire pas cette variable : télécharge le fichier manquant sur le nœud de login.
- Fixe une durée de job (`--time`) large et enregistre des points de sauvegarde. À la fin du temps, le job est tué sans prévenir.
- Jupyter-JSC consomme le budget de calcul tant que la session reste ouverte : ferme-la.
