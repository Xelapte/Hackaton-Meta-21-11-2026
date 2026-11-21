# Dépôt de l'équipe Epi84 — European AI Championship, Paris (21–22/11)

Agents : lisez d'abord `CLAUDE.md`. Humains : fiches 9 (tuto), 11 (rejoindre l'équipe), 12 (scripts), 6 à 8 (tâches).

```
CLAUDE.md                 guide des agents (règles, contexte, cycle de travail)
docs/TACHE_1_watermark.md       tâche 1 : localiser le tatouage (mIoU)
docs/TACHE_2_model_equality.md  tâche 2 : 33 endpoints d'API, familles et modèles (MCC)
docs/TACHE_3_unlearning.md      tâche 3 : désapprendre 3 500 images d'un ViT
docs/CLUSTER.md, STRATEGIE.md, ETAT.md   machines et files ; stratégie ; état courant (à tenir à jour)

watermark/baseline.py     tâche 1 : --mode window (décodage par fenêtres) et --mode unet (segmentation)
met/analyse.py            tâche 2 : similarité entre endpoints, familles/modèles, CSV de soumission
met/mmd_test.py           test MMD du papier (si plusieurs réponses par prompt)
unlearning/unlearn.py     tâche 3 : distillation vers des prédictions adoucies + ancre L2

slurm/env.sh              variables communes : DIR (files, clones, scores), OFFICIEL (données, venvs)
slurm/soumettre.sh        déposer une commande dans la file d'un GPU (--venv unlearning|met|watermark)
slurm/file.sh             état des 4 files ; slurm/annuler.sh : arrêter la commande en cours
slurm/noeud.sh            job du titulaire : garde le nœud, une file par GPU (worker.sh)
slurm/relancer.sh         titulaire : programme la relève du job
slurm/test.sh, shell.sh   test de fumée ; terminal de secours (titulaire)
setup/ajouter_membre.sh   ouvrir le dossier d'équipe à un membre
setup/verif_partage.sh    chaque membre : vérifier ses accès (PARTAGE OK)
setup/partage.sh, fermer.sh  créer/fermer un dossier d'équipe (déjà fait)
setup/nettoyer_depot.sh   une fois par clone : retire les anciens fichiers passés dans archive/
common/scores.py          journal des scores partagé : python -m common.scores
common/dashboard.py       tableau de bord HTML des scores

docs/references/          énoncés officiels des tâches 1 et 2, article MET (2410.20247)
watermark/README.md, met/README.md   index des scripts de chaque tâche (retenu / outil / rejeté)

archive/                  code de préparation écrit d'après les papiers (avant de connaître les vraies
                          tâches) : ne plus l'utiliser, voir archive/README.md
```

Données et environnements : dossier officiel `$OFFICIEL` (`/p/scratch/training2666/huang25/Epi84`),
créé par `hackathon_setup.sh` des organisateurs. Les trois baselines ont été testées sur données
factices (`--fake`) ; elles n'ont pas encore tourné sur les vraies données.
