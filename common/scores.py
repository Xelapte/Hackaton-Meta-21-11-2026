"""Journal des scores : chaque run ajoute une ligne au journal partagé (SCORES_CSV).

Utilisation dans un script :
    from common.scores import log_score
    log_score("scrub", {"lr": 5e-4, "max_steps": 2}, score=0.83, notes="seed 0")

En ligne de commande, pour voir le meilleur score de chaque tâche :
    python -m common.scores                 # plus haut = meilleur
    python -m common.scores --bas met       # pour une tâche où plus bas = meilleur
"""
import argparse
import csv
import datetime
import fcntl
import json
import os
import pathlib
import subprocess

# Journal partagé par toute l'équipe : SCORES_CSV est fixé dans slurm/env.sh
# (<dossier_projet>/scores.csv). Sans cette variable : results/scores.csv du clone.
CSV = pathlib.Path(os.environ.get("SCORES_CSV") or
                   pathlib.Path(__file__).resolve().parents[1] / "results" / "scores.csv")
FIELDS = ["heure", "tache", "config", "score", "commit", "notes"]


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "sans-git"


def log_score(tache, config, score, notes=""):
    """Ajoute une ligne ; un verrou évite que deux GPU écrivent en même temps."""
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)  # la taille peut avoir changé avant le verrou
            if fh.tell() == 0:
                csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
            csv.DictWriter(fh, fieldnames=FIELDS).writerow({
                "heure": datetime.datetime.now().isoformat(timespec="seconds"),
                "tache": tache,
                "config": json.dumps(config, ensure_ascii=False, sort_keys=True),
                "score": score,
                "commit": _commit(),
                "notes": notes,
            })
            fh.flush()  # écrire sur disque AVANT de rendre le verrou
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def meilleurs(taches_plus_bas_mieux=()):
    """Renvoie {tâche: ligne du meilleur run}."""
    if not CSV.exists():
        return {}
    best = {}
    with CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                s = float(row["score"])
            except ValueError:
                continue
            t = row["tache"]
            bas = t in taches_plus_bas_mieux
            if t not in best or (s < float(best[t]["score"]) if bas
                                 else s > float(best[t]["score"])):
                best[t] = row
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", nargs="*", default=[],
                    help="tâches pour lesquelles un score plus bas est meilleur")
    args = ap.parse_args()
    for tache, row in sorted(meilleurs(args.bas).items()):
        print(f"{tache:12s} {row['score']:>10s}  {row['heure']}  commit {row['commit']}"
              f"  {row['config']}")
