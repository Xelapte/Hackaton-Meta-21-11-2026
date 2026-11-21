# Kit pré-hackathon (v2)

Dépôt d'équipe pour le hackathon : trois tâches obligatoires, chacune notée par les tests du jury.

| Tâche | Papier | Dossier | GPU |
|---|---|---|---|
| Tatouage d'images | Pixel Seal (arXiv 2512.16874) | `pixelseal/` | 2 |
| Audit d'API de modèles de langage | Model Equality Testing (arXiv 2410.20247) | `met/` | 0 |
| Désapprentissage (machine unlearning) | SCRUB (arXiv 2302.09880) | `scrub/` | 1 |

Le GPU 3 est un « joker » : l'équipe le réattribue selon les scores, puis le garde pour la démo.

---

## La règle n°1 : tout se fait sur le nœud de login, rien en local

**On n'installe et on ne lance RIEN sur nos ordinateurs personnels** : ni `pip install`, ni modèle, ni poids, ni données, ni script Python.

| Où | Ce qu'on y fait |
|---|---|
| **Ordinateur perso** | Uniquement `ssh jsc` pour se connecter. On peut aussi y éditer le code puis faire `git push`, rien de plus. |
| **Nœud de login** (`ssh jsc`) | **Tout le reste** : ton clone du dépôt (dans `$DIR/membres/<identifiant>/`), `git pull`, installations, téléchargements (`pip`, `wget`, `hf download`, jeux de données), et le dépôt de tes commandes dans les files des GPU (`slurm/soumettre.sh`). |
| **Nœuds GPU** | L'exécution des commandes des files, dans le job du coordinateur. On ne s'y connecte pas. **Ils n'ont pas internet** : un téléchargement y fait planter le script. |

Ne lance pas non plus de calcul lourd directement sur le nœud de login : il est partagé avec tout le centre. Tout ce qui utilise un GPU passe par les files d'attente (`slurm/soumettre.sh`).

### Qui lance les jobs : le coordinateur du nœud

- **Un seul membre, le coordinateur du nœud**, lance le seul job : `sbatch slurm/noeud.sh`. Ce job garde le nœud (4 GPU) et ouvre **une file d'attente par GPU**. Un deuxième job consommerait un deuxième nœud et doublerait le budget. Le coordinateur le relance avant la fin de son `--time`.
- **Chaque membre a son propre compte et sa propre clé SSH.** Slurm ne laisse entrer dans un job que la personne qui l'a lancé : les autres ne se connectent donc jamais au nœud GPU. Ils **déposent leurs commandes** dans la file de leur GPU depuis le nœud de login, et la file les exécute une par une.
- **Le suppléant** prend le relais quand le coordinateur dort ou ne répond pas : si le job s'arrête, il relance `slurm/noeud.sh` sous son propre compte. Les commandes en attente ne sont pas perdues.
- **Les petits tests sont libres** : une commande de moins de 30 minutes, dans la file du GPU de sa tâche, dont on suit le journal.
- **Tout le reste demande l'accord du coordinateur** (ou du suppléant la nuit) : run long ou de nuit, autre GPU ou joker, annulation de la commande de quelqu'un d'autre, job Slurm, gros téléchargement, soumission officielle. Le coordinateur tient à jour le tableau « Ce qui tourne » de `docs/ETAT.md`.

```
ordinateur --ssh jsc--> nœud de login --dépose--> $DIR/queue/gpuN/attente/ --exécute--> nœud GPU, GPU N
                         (ton clone)                (dossier partagé)           (job slurm/noeud.sh du coordinateur)
```

Les gros fichiers (`env/`, `models/`, `data/`, `ckpts/`, `repos/`, `hf_cache/`) vivent dans le dossier projet du cluster (`$DIR`), jamais dans le dépôt. On y trouve aussi les clones des membres (`$DIR/membres/`), les files d'attente (`$DIR/queue/`) et **le journal des scores partagé** (`$DIR/scores.csv`).

---

## L'installation (sur le nœud de login)

**Une seule fois pour l'équipe, par le coordinateur**, depuis la racine de son clone :

```bash
ssh jsc
bash setup/install_env.sh <dossier_projet>          # 1. environnement Python
source <dossier_projet>/env/bin/activate            # 2. l'activer
hf auth login                                       #    si le modèle demande un accès (Llama, Mistral)
bash setup/download_all.sh <dossier_projet> <modele_hf>   # 3. dépôts, poids, CIFAR, modèle
bash setup/install_pixelseal.sh <dossier_projet>    # 4. bibliothèque videoseal
bash setup/partage.sh <dossier_projet>              # 5. files d'attente et droits du groupe
# 6. remplir slurm/env.sh et les lignes #SBATCH de slurm/test.sh et slurm/noeud.sh
sbatch slurm/test.sh                                # 7. test de fumée : lire logs/test-<id>.out
sbatch slurm/noeud.sh                               # 8. garde le nœud, ouvre les 4 files
```

**Chaque membre, une fois**, sur le nœud de login :

```bash
echo 'umask 002' >> ~/.bashrc                       # fichiers modifiables par toute l'équipe
mkdir -p <dossier_projet>/membres/$USER && cd <dossier_projet>/membres/$USER
git clone git@github.com:Xelapte/Hackaton-Meta-21-11-2026.git && cd Hackaton-Meta-21-11-2026
ln -s <dossier_projet>/ckpts ckpts                  # lien vers les poids de Pixel Seal
# remplir slurm/env.sh (mêmes valeurs que le coordinateur)
```

**À chaque run**, depuis la racine de ton clone :

```bash
bash slurm/soumettre.sh <gpu> <nom_court> <commande...>   # ex. : bash slurm/soumettre.sh 1 scrub_base bash scrub/run.sh
bash slurm/file.sh                                        # état des 4 files
tail -f $DIR/queue/logs/<nom>.log                          # suivre le run
bash slurm/annuler.sh <gpu>                               # arrêter la commande en cours
source slurm/env.sh && python -m common.scores            # meilleurs scores de l'équipe
```

Une file exécute une commande à la fin de la précédente. Pour plusieurs runs en parallèle sur un GPU, mets-les dans un même script avec `&` puis `wait`. Ne modifie pas un script en attente ou en cours : il est lu au démarrage.

---

## Le squelette, tâche par tâche

### Pixel Seal : tatouage d'images (GPU 2)

L'objectif : cacher un message de 256 bits dans une image, puis le relire même après des attaques (JPEG, recadrage, flou…). Le compromis : un tatouage plus fort est plus robuste mais plus visible.

| Fichier | Rôle |
|---|---|
| `pixelseal/smoke_test.py` | Le test de fumée de la tâche : vérifie que les poids sont lus dans `ckpts/` sans téléchargement, et que le message est relu sur une image non attaquée. `bash slurm/soumettre.sh 2 smoke python pixelseal/smoke_test.py`. |
| `pixelseal/quickstart.py` | La baseline. Tatoue N images, applique 6 attaques, relit le message, puis calcule le PSNR (qualité d'image), le taux de bits justes et la p-value. Le score est le taux moyen de bits justes sous attaque. À adapter à la métrique du jury. |
| `pixelseal/run.sh` | La baseline complète, à déposer dans la file du GPU 2 : `quickstart.py` avec trois forces de tatouage (`--scaling_w` 0,2, 0,3 et 0,4) en parallèle. |
| `pixelseal/CLAUDE.md` | Le guide de la tâche : principe, API de videoseal, pièges et leviers d'amélioration. |

Il a besoin de `IMAGES` (le dossier des images de test, connu à 12h) et du lien `ckpts` à la racine du dépôt.

### MET : audit d'API de modèles de langage (GPU 0)

L'objectif : savoir si une API sert vraiment le modèle annoncé, ou une version modifiée (quantifiée, tatouée…). On utilise un test statistique à deux échantillons : la MMD avec un noyau de Hamming, et une p-value obtenue par permutation.

| Fichier | Rôle |
|---|---|
| `met/mmd_test.py` | Le cœur statistique : le noyau de Hamming, la MMD sans biais calculée prompt par prompt, et le test de permutation `mmd_test(...)`. Testé : 5 % de faux positifs sous H0. |
| `met/generate_vllm.py` | Génère les réponses de référence (P) avec vLLM, en fp32, fp16 ou bf16. Une graine différente est obligatoire pour chaque fichier. |
| `met/generate_hf.py` | Génère les variantes à détecter (Q) avec transformers : `int8`, `nf4`, `watermark`, `fp16` ou `fp32`. |
| `met/compare.py` | Compare deux échantillons (`.npz` de tokens ou `.json` de textes) et écrit la p-value comme score. Plus bas = différence détectée. |
| `met/prompts_exemple.jsonl` | Le format des prompts : une ligne `{"id": "...", "text": "..."}` par prompt. |
| `met/run.sh` | La baseline complète, à déposer dans la file du GPU 0 : P et P' (contrôle), Q pour nf4, int8 et watermark, puis toutes les comparaisons. |
| `met/CLAUDE.md` | Le guide de la tâche : principe, réglages du papier et pièges (graines, gabarit de chat, décodage, mémoire). |

Il a besoin de `MODEL_PATH` (le modèle téléchargé par `download_all.sh`). On lit le meilleur score avec `python -m common.scores --bas met`.

### SCRUB : désapprentissage (GPU 1)

L'objectif : faire **oublier** un sous-ensemble de données (le forget set) à un modèle déjà entraîné, sans tout réentraîner, et en gardant ses performances sur le reste. On utilise un schéma enseignant–élève : l'élève s'éloigne de l'enseignant sur le forget set et reste proche de lui sur le retain set.

| Fichier | Rôle |
|---|---|
| `scrub/cifar.py` | La boîte à outils : ResNet-18 adapté à CIFAR, chargement des données (`--fake` pour tester sans données), découpage forget/retain, entraînement et évaluation. |
| `scrub/scrub.py` | L'algorithme : SCRUB, SCRUB+R (rembobinage vers le point de sauvegarde le plus proche de la référence) et l'attaque d'appartenance (MIA). |
| `scrub/train_original.py` | Entraîne le modèle original, ou le modèle réentraîné sans le forget set (`--retrain`), qui sert de référence. |
| `scrub/run_scrub.py` | Applique SCRUB et SCRUB+R, mesure les erreurs (test, retain, forget) et la MIA, puis écrit le score : l'erreur sur le forget set. À adapter à la métrique du jury. |
| `scrub/run.sh` | La baseline complète, à déposer dans la file du GPU 1 : l'original et le réentraînement en parallèle, puis SCRUB en mode classe et en mode sélectif. |
| `scrub/CLAUDE.md` | Le guide de la tâche : principe, réglages du papier (tableau 3), pièges et leviers. |

Il a besoin de `DIR` (CIFAR doit déjà être dans `$DIR/data`).

### Commun aux trois tâches

| Fichier | Rôle |
|---|---|
| `common/scores.py` | Le journal des scores. `log_score(tache, config, score, notes)` ajoute une ligne au journal partagé `$DIR/scores.csv` (variable `SCORES_CSV`, fixée dans `slurm/env.sh`). `source slurm/env.sh && python -m common.scores` affiche le meilleur score de chaque tâche. **Un résultat absent du CSV n'existe pas pour l'équipe.** |
| `common/dashboard.py` | Transforme le journal des scores en une page HTML statique : `python -m common.dashboard` écrit `results/dashboard.html`. |
| `common/smoke_test.py` | Le test de fumée GPU : GPU visibles, bf16, vitesse, imports, présence du modèle. Lancé par `slurm/test.sh`. |

### Installation : `setup/` (nœud de login uniquement)

| Fichier | Rôle |
|---|---|
| `setup/install_env.sh` | Crée l'environnement Python de l'équipe dans `$DIR/env` (vLLM, torch, transformers…). |
| `setup/download_all.sh` | Télécharge les trois dépôts officiels, les poids de Pixel Seal, CIFAR-10 et CIFAR-100, et le modèle de langage. |
| `setup/install_pixelseal.sh` | Installe la bibliothèque videoseal, qui contient Pixel Seal, dans l'environnement. À lancer après `download_all.sh`. |
| `setup/partage.sh` | Crée les files d'attente des 4 GPU dans `$DIR/queue/` et ouvre le dossier projet au groupe de l'équipe. Une fois, par le coordinateur. |

### Lancement sur les GPU : `slurm/` (les files d'attente)

| Fichier | Rôle |
|---|---|
| `slurm/env.sh` | Les variables communes (`DIR`, `MODEL_PATH`, `IMAGES`, `SCORES_CSV`, mode hors ligne `HF_HUB_OFFLINE=1`, `umask 002`). **À remplir une fois.** |
| `slurm/test.sh` | Un job de test de 10 minutes : `nvidia-smi` puis `smoke_test.py`. À lancer en premier, par le coordinateur. |
| `slurm/noeud.sh` | **Le job du coordinateur** : garde le nœud et lance un `worker.sh` par GPU. Arrêt propre : `touch $DIR/queue/STOP`. |
| `slurm/worker.sh` | Exécute une par une, dans l'ordre de dépôt, les commandes de la file d'un GPU (`attente/` → `en_cours/` → `fini/`). Lancé par `noeud.sh`, jamais à la main. |
| `slurm/soumettre.sh` | **Ce que chaque membre utilise** : dépose une commande dans la file d'un GPU, depuis son clone. Elle s'exécute avec `slurm/env.sh` chargé et `CUDA_VISIBLE_DEVICES` fixé. |
| `slurm/file.sh` | Affiche l'état des 4 files : active ou non, commande en cours, en attente, dernières finies. |
| `slurm/annuler.sh` | Arrête la commande en cours sur un GPU (la file passe à la suivante). Pour retirer une commande en attente : `rm $DIR/queue/gpuN/attente/<nom>.sh`. |
| `slurm/vague.sh` | **Secours**, si les files ne marchent pas. Une « vague » : un processus par GPU en parallèle (GPU 0 : `met/run.sh`, GPU 1 : `scrub/run.sh`, GPU 2 : `pixelseal/run.sh`, GPU 3 : le joker si `JOKER` est rempli). |
| `slurm/shell.sh` | **Secours, pour le coordinateur seulement.** Un terminal interactif dans la réservation, limité à un seul GPU : `bash slurm/shell.sh <jobid> <gpu>`. Pratique pour déboguer. Le `<jobid>` est celui de la réservation du coordinateur. Non testé. |

### Sorties et documentation

| Fichier ou dossier | Rôle |
|---|---|
| `results/` | Sorties locales du clone, dont `dashboard.html`. Le journal des scores n'y est plus : il est partagé dans `$DIR/scores.csv`. Les grosses sorties (`results/met/`, `results/scrub/`) ne sont pas versionnées. |
| `logs/` | Les sorties des jobs Slurm (non versionnées). |
| `CLAUDE.md` | Le guide général pour les agents et les membres : règles absolues, cycle de travail, style. |
| `docs/CLUSTER.md` | Les machines, la connexion SSH, Slurm, les règles du GPU partagé. |
| `docs/STRATEGIE.md` | Le planning des 24 h, les principes de score, le fil conducteur du pitch. |
| `docs/ETAT.md` | L'état courant du travail (meilleurs scores, ce qui tourne, à faire). **À lire en arrivant, à mettre à jour en partant.** |

---

## Ce qui a été testé

- **Testé sans GPU :** les files d'attente (dépôt, ordre, annulation, arrêt, reprise), les scores, le tableau de bord, la MMD (`met/compare.py`) et toute la chaîne SCRUB sur données factices (`--fake`).
- **Vérifié à blanc :** l'enchaînement des `run.sh` et les gardes sur les variables.
- **Jamais exécuté** (ni GPU ni poids disponibles pendant la préparation) : `pixelseal/quickstart.py`, `met/generate_vllm.py`, `met/generate_hf.py` et `slurm/*.sh`. Au premier échec, lis l'erreur avant de réécrire quoi que ce soit.
