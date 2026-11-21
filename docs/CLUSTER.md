# Le cluster : machines, Slurm, GPU

Les GPU sont au centre de calcul de Jülich (JSC). On y accède par JuDoor. Les noms exacts (machine, projet, partition, réservation) sont donnés par les organisateurs : ils remplacent les valeurs entre chevrons.

## Les trois machines

| Machine | Internet | GPU | Rôle |
|---|---|---|---|
| Ordinateur d'un membre | oui | non | Terminal SSH, édition du code, `git push`. **Rien d'installé, rien de téléchargé.** |
| Nœud de login (`ssh jsc`) | oui | non | **Tous** les téléchargements et installations, `git`, `sbatch`. Pas de calcul lourd. |
| Nœud GPU | **non** | 4 × 40 Go | Exécution, via Slurm uniquement. |

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
```

## L'installation, sur le nœud de login uniquement

```bash
bash setup/install_env.sh <dossier_projet>
source <dossier_projet>/env/bin/activate
hf auth login                                       # accès Llama ou Mistral accordé sur le site
bash setup/download_all.sh <dossier_projet> <modele_hf>
bash setup/install_pixelseal.sh <dossier_projet>
ln -s <dossier_projet>/ckpts ckpts                  # à la racine du dépôt
# puis remplir slurm/env.sh et les lignes #SBATCH de slurm/test.sh et slurm/vague.sh
sbatch slurm/test.sh                                # test de fumée : regarder logs/test-<id>.out
```

## Lancer sur les GPU

On ne peut pas réserver moins d'un nœud entier, soit 4 GPU. Un seul membre, le « coordinateur du nœud », réserve, et les autres se partagent sa réservation. Il lance `salloc` depuis une session `tmux` : si sa session tombe, la réservation tombe avec elle.

**La règle.** Les petits tests sont libres : sur ton propre GPU, dans la réservation ouverte, en moins de 30 minutes. Tout le reste se fait avec l'accord du coordinateur du nœud, ou de son suppléant la nuit : run long ou de nuit, vague, nouveau job, autre GPU ou joker, annulation, gros téléchargement, soumission officielle. Il tient à jour le tableau « Ce qui tourne » de `docs/ETAT.md`.

**Méthode 1 : les vagues.** C'est la plus robuste.

```bash
sbatch slurm/vague.sh      # GPU 0 : met/run.sh, GPU 1 : scrub/run.sh, GPU 2 : pixelseal/run.sh
squeue -A <projet>         # état du job
tail -f logs/gpu1-<jobid>.log
```

- Pour changer d'expérience, modifie `<tache>/run.sh`.
- Pour utiliser le GPU 3, copie un `run.sh` en changeant `OUT=…`, puis mets son chemin dans la variable `JOKER` de `vague.sh`.

**Méthode 2 : un terminal dans la réservation.** C'est la plus pratique pour déboguer.

```bash
bash slurm/shell.sh <jobid> <gpu>   # srun --overlap, CUDA_VISIBLE_DEVICES=<gpu>
source slurm/env.sh
python -c "import torch; print(torch.cuda.device_count())"   # doit afficher 1
```

`slurm/shell.sh` n'a pas été testé. Si Slurm refuse `--overlap`, repasse aux vagues.

Les partitions : `develbooster` pour les tests courts, `booster` pour les longs. Ce sont les noms sur JUWELS Booster : adapte-les aux consignes des organisateurs.

## Les règles du GPU partagé

- Fixe toujours `CUDA_VISIBLE_DEVICES`. Dans le code, utilise `"cuda"` : le GPU choisi s'appelle `cuda:0`.
- `nvidia-smi` ignore `CUDA_VISIBLE_DEVICES` et montre les quatre GPU. C'est utile pour voir qui utilise quoi, pas pour vérifier ce que voit ton programme.
- vLLM réserve 90 % de la mémoire du GPU par défaut (`--gpu_mem 0.90`). Pour partager un GPU avec un autre processus, baisse cette valeur.
- Les cœurs du processeur et la mémoire vive sont partagés aussi : `nproc` donne le nombre de cœurs. Limite le nombre de processus de chargement de données.
- `slurm/env.sh` fixe `HF_HUB_OFFLINE=1`. Si un script essaie quand même de télécharger quelque chose, ne retire pas cette variable : télécharge le fichier manquant sur le nœud de login.
- Fixe une durée de job (`--time`) large et enregistre des points de sauvegarde. À la fin du temps, le job est tué sans prévenir.
- Jupyter-JSC consomme le budget de calcul tant que la session reste ouverte : ferme-la.
