# Kit pré-hackathon (v2)

Squelette de dépôt d'équipe et scripts de préparation (fiches 5 à 8).

```
CLAUDE.md                 guide pour les agents (lire en premier), + un CLAUDE.md par tâche
docs/                     CLUSTER.md, STRATEGIE.md, ETAT.md (état courant, à tenir à jour)
common/smoke_test.py      test de fumée GPU (lancé par slurm/test.sh)
common/scores.py          journal des scores : log_score(...) et python -m common.scores
common/dashboard.py       page HTML statique des scores : python -m common.dashboard
setup/install_env.sh      environnement Python de l'équipe (nœud de login)
setup/download_all.sh     dépôts, poids Pixel Seal, CIFAR-10/100, modèle de langage
setup/install_pixelseal.sh bibliothèque videoseal (après download_all.sh)
slurm/env.sh              variables communes (dossier projet, modèle, images) à remplir
slurm/test.sh             job de test de 10 minutes
slurm/vague.sh            un processus par GPU : lance met/, scrub/, pixelseal/run.sh
slurm/shell.sh            terminal interactif limité à un GPU, dans une réservation
pixelseal/quickstart.py   baseline Pixel Seal : tatouer, attaquer, relire, score
pixelseal/run.sh          baseline lancée sur le GPU 2 (trois forces en parallèle)
met/mmd_test.py           test MMD-Hamming (fiche 2)
met/generate_vllm.py      réponses de référence (fp32) ou fp16/bf16 avec vLLM
met/generate_hf.py        variantes int8, nf4 et tatouée avec transformers
met/compare.py            compare deux échantillons, écrit le score
met/prompts_exemple.jsonl prompts d'exemple au format attendu
met/run.sh                pipeline complet lancé sur le GPU 0
scrub/scrub.py            SCRUB, SCRUB+R, attaque d'appartenance (fiche 3)
scrub/cifar.py            ResNet-18 pour CIFAR, données, entraînement, évaluation
scrub/train_original.py   modèle original et réentraînement de référence
scrub/run_scrub.py        SCRUB + SCRUB+R + mesures, écrit le score
scrub/run.sh              pipeline complet lancé sur le GPU 1
logs/ results/            sorties des jobs, scores.csv et dashboard.html
```

Ordre de préparation (nœud de login, depuis la racine du dépôt) :

1. `bash setup/install_env.sh <dossier_projet>`
2. `source <dossier_projet>/env/bin/activate`
3. `bash setup/download_all.sh <dossier_projet> <modele_hugging_face>`
4. `bash setup/install_pixelseal.sh <dossier_projet>`
5. `ln -s <dossier_projet>/ckpts ckpts` (poids de Pixel Seal)
6. Remplir `slurm/env.sh` et les lignes `#SBATCH` de `slurm/test.sh` et `slurm/vague.sh`,
   puis `sbatch slurm/test.sh`

Testé sans GPU : scores, tableau de bord, MMD (met/compare.py), chaîne SCRUB complète
sur données factices (`--fake`). Non exécutés ici (ni GPU, ni accès aux poids) :
pixelseal/quickstart.py, met/generate_vllm.py, met/generate_hf.py, slurm/*.sh.
Les `run.sh` ont été vérifiés à blanc (enchaînement des commandes, gardes sur les
variables), sans GPU.
