"""v80 : sondes d'identite (prompts5.jsonl), reponses courtes (64 tokens). Usage : python ident5.py <endpoints> <repeats> <sortie.jsonl>"""
import os, sys, json
os.environ["TEAM_API_KEY"] = open(os.path.expanduser("~/.hack_api_key")).read().strip()
sys.path.insert(0, "/p/scratch/training2666/huang25/Epi84/model_equality")
from pathlib import Path
import query_reference as Q
V = Path("/p/scratch/training2666/huang25/Epi84/unlearning/v80")
prompts = [(r["id"], r["prompt"]) for r in (json.loads(l) for l in open(V / "prompts5.jsonl"))]
Q.query(prompts, output=V / sys.argv[3], endpoints=sys.argv[1].split(","), repeats=int(sys.argv[2]), max_tokens=64)
