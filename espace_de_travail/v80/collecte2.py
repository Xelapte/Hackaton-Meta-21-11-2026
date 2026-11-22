"""v80 : collecte ciblee tache 2 (autorisee par E. le 22/11 ~10h20 : 70 requetes restantes par endpoint).
Reutilise query() du script officiel query_reference.py (inchange). Sortie : v80/collecte2.jsonl (ajout, reprise sure).
Usage : python collecte2.py ep-13,ep-19 6"""
import os, sys, json
os.environ["TEAM_API_KEY"] = open(os.path.expanduser("~/.hack_api_key")).read().strip()
sys.path.insert(0, "/p/scratch/training2666/huang25/Epi84/model_equality")
from pathlib import Path
import query_reference as Q
V = Path("/p/scratch/training2666/huang25/Epi84/unlearning/v80")
prompts = [(r["id"], r["prompt"]) for r in (json.loads(l) for l in open(V / "prompts2.jsonl"))]
eps = sys.argv[1].split(","); rep = int(sys.argv[2])
Q.query(prompts, output=V / "collecte2.jsonl", endpoints=eps, repeats=rep, max_tokens=256)
