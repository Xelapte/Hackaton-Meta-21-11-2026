# Le cluster : machines, Slurm, GPU

Les GPU sont au centre de calcul de Jülich (JSC). On y accède par JuDoor. Les noms exacts (machine, projet, partition, réservation) sont donnés par les organisateurs : ils remplacent les valeurs entre chevrons.

## Les trois machines

| Machine | Internet | GPU | Rôle |
|---|---|---|---|
| Ordinateur d'un membre | oui | non | Terminal SSH, édition du code, `git push`. **Rien d'installé, rien de téléchargé.** |
| Nœud de login (`ssh jsc`) | oui | non | **Tous** les téléchargements et installations, `git`, dépôt des commandes dans les files. Pas de calcul lourd. |
| Nœud GPU | **non** | 4 × 40 Go | Exécution des commandes des files, dans le job du coordinateur. On ne s'y connecte pas. |

Le nœud de login et le nœud GPU voient le même dossier projet, `$DIR`, placé sur un espace partagé comme `$PROJECT` ou `$SCRATCH` (à confirmer avec les organisateurs). Ce qui est téléchargé sur le nœud de login est donc lu tel quel sur le nœud GPU.

## La connexion

On utilise une seule clé SSH, créée sur l'ordinateur du membre : `~/.ssh/id_ed25519-JSC`. Sa partie publique est déposée dans JuDoor, avec une clause `from="…"` qui limite les adresses IP autorisées. L'alias `jsc` est défini dans `~/.ssh/config` : `ssh jsc`, `scp fichier jsc:`.

Si la connexion est refusée, c'est presque toujours parce que l'adresse IP a changé, par exemple sur le Wi-Fi du lieu. Il faut alors ajouter la plage d'adresses dans JuDoor.

L'accès à GitHub depuis le nœud de login se fait avec une **autre** clé, créée sur le nœud de login et déposée sur GitHub.

## Le dossier projet ($DIR)

```
$DIR/env/        environnement Python de l'équipe (setup/install_env.sh)
$DIR/repos/      dépôts officiels : videoseal, model-equality-testing, SCRUB
$DIR/ckpts/      poids de Pixel Seal (pixelseal_checkpoint.pth, videoseal_y_256b_img.pth)
$DIR/models/     modèle de langage pour MET (hf download)
$DIR/data/       CIFAR-10 et CIFAR-100
$DIR/hf_cache/   cache Hugging Face (HF_HOME)
$DIR/membres/    un clone du dépôt par membre
$DIR/queue/      files d'attente des GPU (gpu0 … gpu3) et leurs journaux (logs/)
$DIR/scores.csv  journal des scores de toute l'équipe
```

## L'installation, sur le nœud de login uniquement

```bash
bash setup/install_env.sh <dossier_projet>
source <dossier_projet>/env/bin/activate
hf auth login                                       # accès Llama ou Mistral accordé sur le site
bash setup/download_all.sh <dossier_projet> <modele_hf>
bash setup/install_pixelseal.sh <dossier_projet>
ln -s <dossier_projet>/ckpts ckpts                  # à la racine du dépôt
bash setup/partage.sh <dossier_projet>              # files d'attente et droits du groupe
# puis remplir slurm/env.sh et les lignes #SBATCH de slurm/test.sh et slurm/noeud.sh
sbatch slurm/test.sh                                # coordinateur : test de fumée (logs/test-<id>.out)
sbatch slurm/noeud.sh                               # coordinateur : garde le nœud, ouvre les files
```

## Lancer sur les GPU : les files d'attente

Chaque membre a **son propre compte et sa propre clé SSH** ; on ne partage jamais une clé. Slurm ne laisse entrer dans un job que la personne qui l'a lancé. Les autres membres ne peuvent donc pas ouvrir de terminal sur le nœud du coordinateur. D'où ce fonctionnement :

```
ordinateur --ssh jsc--> nœud de login --dépose--> $DIR/queue/gpuN/attente/ --exécute--> nœud de calcul, GPU N
                         (votre clone)              (dossier partagé)           (job slurm/noeud.sh du coordinateur)
```

**Une fois.**

- Le coordinateur lance `bash setup/partage.sh <dossier_projet>`, qui crée les files et ouvre le dossier au groupe du projet.
- Chaque membre ajoute `umask 002` à son `~/.bashrc` et clone le dépôt dans `$DIR/membres/<identifiant>/`.
- Le coordinateur lance `sbatch slurm/noeud.sh` : un seul job, qui garde le nœud et exécute une file par GPU. Il le relance avant la fin de `--time`. Une commande coupée par la fin du job est classée « interrompue » ; les commandes en attente ne sont pas perdues.

**À chaque run**, depuis la racine de ton clone, sur le nœud de login :

```bash
bash slurm/soumettre.sh <gpu> <nom_court> <commande...>   # ex. : 1 scrub_lr bash scrub/run.sh
bash slurm/file.sh                                        # état des 4 files
tail -f $DIR/queue/logs/<nom>.log                          # suivre le run
bash slurm/annuler.sh <gpu>                               # arrêter la commande en cours
touch $DIR/queue/STOP                                     # coordinateur : fermer les files proprement
```

Le fonctionnement des files :

- Une file exécute une commande à la fois, dans l'ordre de dépôt, avec `slurm/env.sh` chargé et `CUDA_VISIBLE_DEVICES` fixé.
- Pour plusieurs runs en parallèle sur un GPU, mets-les dans un même script avec `&` puis `wait`.
- La commande lit les fichiers au démarrage, pas au dépôt. Ne modifie pas un script en attente ou en cours.
- Le journal des scores est partagé : `SCORES_CSV=$DIR/scores.csv`, fixé dans `slurm/env.sh`.

**La règle.** Les petits tests sont libres : dans la file de ton propre GPU, en moins de 30 minutes, en suivant le journal. Tout le reste se fait avec l'accord du coordinateur du nœud, ou de son suppléant la nuit : run long ou de nuit, autre GPU ou joker, annulation de la commande d'un autre, job Slurm, gros téléchargement, soumission officielle.

Les files ont été testées sans GPU : dépôt, ordre, annulation, arrêt, reprise après interruption. Elles restent à valider sur la vraie machine. Les solutions de secours :

- `sbatch slurm/vague.sh`, lancé par le coordinateur, exécute le `run.sh` de chaque tâche ;
- `slurm/shell.sh`, pour le coordinateur seulement, ouvre un terminal dans son propre job.

Les partitions : `develbooster` pour les tests courts, `booster` pour les longs. Ce sont les noms sur JUWELS Booster : adapte-les aux consignes des organisateurs.

## Les règles du GPU partagé

- Fixe toujours `CUDA_VISIBLE_DEVICES`. Dans le code, utilise `"cuda"` : le GPU choisi s'appelle `cuda:0`.
- `nvidia-smi` ignore `CUDA_VISIBLE_DEVICES` et montre les quatre GPU. C'est utile pour voir qui utilise quoi, pas pour vérifier ce que voit ton programme.
- vLLM réserve 90 % de la mémoire du GPU par défaut (`--gpu_mem 0.90`). Pour partager un GPU avec un autre processus, baisse cette valeur.
- Les cœurs du processeur et la mémoire vive sont partagés aussi : `nproc` donne le nombre de cœurs. Limite le nombre de processus de chargement de données.
- `slurm/env.sh` fixe `HF_HUB_OFFLINE=1`. Si un script essaie quand même de télécharger quelque chose, ne retire pas cette variable : télécharge le fichier manquant sur le nœud de login.
- Fixe une durée de job (`--time`) large et enregistre des points de sauvegarde. À la fin du temps, le job est tué sans prévenir.
- Jupyter-JSC consomme le budget de calcul tant que la session reste ouverte : ferme-la.
