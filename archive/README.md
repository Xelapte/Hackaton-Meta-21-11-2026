# Archive : code de préparation d'après les papiers

Écrit avant de connaître les vraies tâches, à partir des papiers Pixel Seal, Model Equality
Testing et SCRUB. Les vraies tâches sont différentes (voir `docs/TACHE_*.md`) : **ne plus
utiliser ce code tel quel**. Il reste consultable pour s'en inspirer.

- `setup/install_env.sh`, `download_all.sh`, `install_pixelseal.sh` : remplacés par
  `hackathon_setup.sh` (organisateurs) et le `download_models.py` de la tâche 1.
- `met/` : génération locale avec vLLM/transformers et comparaison MMD. La tâche réelle passe
  par une API. Le test MMD lui-même est resté dans `met/mmd_test.py`.
- `scrub/` : SCRUB sur CIFAR-10 avec un retain set. Idées réutilisables pour la tâche 3.
- `pixelseal/` : tatouer/attaquer/relire avec Pixel Seal. La tâche 1 utilise `wmloc`.
- `slurm/vague.sh` : lançait les `run.sh` ci-dessus ; les files d'attente le remplacent.
