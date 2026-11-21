# Context
Hackathon (CISPA EU AI Championship) on JURECA-DC, Jülich supercomputer.
Project: training2666. Team folder: /p/scratch/training2666/huang25/Epi84
My task: model_equality. My user: houmache2. My git branch: wael.

# Where things are
- My working copy (the ONLY place you may edit): /p/scratch/training2666/huang25/Epi84/wael-repo
- Shared task folder (READ-ONLY, used by the whole team): /p/scratch/training2666/huang25/Epi84/model_equality
- Python env (shared, do not modify): /p/scratch/training2666/huang25/Epi84/model_equality/.venv

# Hard rules
- Never create, modify, move or delete files outside wael-repo.
- Never touch other team folders (Hackathon_Watermark_Localisation_Task, unlearning) or other users' folders.
- Never pip/uv install into the shared .venv. If a package is missing, ask me first.
- Never run `rm -rf` or delete datasets/models.
- Never commit or push to `main`. Only work on branch `wael`. Never force-push.
- Do not commit large files (models, datasets, checkpoints, outputs). Keep them out of git via .gitignore.
- Never print, log or commit tokens/secrets (GitHub ghp_…, Hugging Face hf_…).

# Cluster rules (JURECA)
- This is a LOGIN node: never run training, inference or GPU code here. Only light scripts (reading files, small tests, plotting).
- Heavy work = write/modify a Slurm script (job.sh) and tell me to run `sbatch job.sh`. Do not submit jobs yourself without asking.
- Compute nodes have NO internet: anything to download (Hugging Face models/datasets) must be downloaded on the login node first, into the shared HF cache already set in the environment.
- Slurm 25.11: if using `--gpus-per-task`, also add `--gres=gpu:X`.
- GPU budget is small and shared by the team: keep jobs short, test on small data first.
- Useful: `squeue -u $USER`, `scancel <id>`, logs in `slurm-<id>.out`.

# How to work with me
- Explain things simply; I'm new to this cluster.
- Before any command that changes something (git push, sbatch, installs, deletes), tell me what it does and wait for my OK.
- Commit small working steps with clear messages.
