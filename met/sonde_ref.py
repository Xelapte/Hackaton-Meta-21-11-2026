"""Tâche 2 : interroger les endpoints de RÉFÉRENCE (ref-01 … ref-09, gratuits et sans quota) avec une batterie de prompts.

Verrou : refuse tout endpoint qui ne commence pas par « ref- » (le budget des ep- n'est jamais touché ici).
La clé d'API est lue dans TEAM_API_KEY (jamais affichée ni écrite). Au plus --parallele requêtes à la fois
(la limite de l'équipe est 4 : on en laisse aux autres membres). Reprise : les (endpoint, id, tirage) déjà
présents dans --sortie ne sont pas redemandés.

  python met/sonde_ref.py --prompts met/batterie.csv --tirages 4 --sortie $OFFICIEL/model_equality/output/huang25/sonde_ref.jsonl
  python met/sonde_ref.py --prompts met/batterie.csv --fake
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://34.147.229.55/33-model-equality/v1/chat/completions"
REFS = [f"ref-{i:02d}" for i in range(1, 10)]


def lire_prompts(chemin):
    with open(chemin, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True, help="CSV : id,type,max_tokens,prompt")
    ap.add_argument("--endpoints", default=",".join(REFS))
    ap.add_argument("--tirages", type=int, default=4)
    ap.add_argument("--parallele", type=int, default=2)
    ap.add_argument("--sortie", default=None)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    eps = args.endpoints.split(",")
    interdits = [e for e in eps if not e.startswith("ref-")]
    if interdits:
        sys.exit(f"Refusé : {interdits} ne sont pas des endpoints de référence (budget ep- protégé).")
    prompts = lire_prompts(args.prompts)
    sortie = Path(args.sortie or "/tmp/sonde_ref_fake.jsonl")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    deja = set()
    if sortie.exists():
        for l in open(sortie, encoding="utf-8"):
            r = json.loads(l); deja.add((r["endpoint"], r["id"], r["tirage"]))
    jobs = [(e, p, t) for p in prompts for t in range(args.tirages) for e in eps if (e, p["id"], t) not in deja]
    print(f"{len(prompts)} prompts x {len(eps)} endpoints x {args.tirages} tirages : {len(jobs)} requêtes à faire "
          f"({len(deja)} déjà faites)", flush=True)
    if args.fake:
        import random
        rnd = random.Random(0)
        with open(sortie, "a", encoding="utf-8") as f:
            for e, p, t in jobs:
                f.write(json.dumps({"endpoint": e, "id": p["id"], "type": p["type"], "tirage": t,
                                    "response": rnd.choice(["42", "43", "Paris"])}, ensure_ascii=False) + "\n")
        print(f"Écrit (faux) : {sortie}"); return
    import requests
    cle = os.environ.get("TEAM_API_KEY", "")
    if not cle:
        sys.exit("TEAM_API_KEY n'est pas définie.")
    verrou = threading.Lock(); fait = [0]; t0 = time.time()
    session = requests.Session(); session.headers["X-API-Key"] = cle

    def un(job):
        e, p, t = job
        for essai in range(6):
            try:
                r = session.post(URL, timeout=120, json={"model": e, "max_tokens": int(p["max_tokens"]),
                                                        "messages": [{"role": "user", "content": p["prompt"]}]})
            except requests.RequestException:
                time.sleep(10); continue
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10))); continue
            if r.status_code >= 500:
                time.sleep(10); continue
            if r.status_code != 200:
                print(f"  {e} {p['id']} : HTTP {r.status_code} {r.text[:120]}", flush=True); return
            txt = r.json()["choices"][0]["message"]["content"]
            with verrou:
                with open(sortie, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"endpoint": e, "id": p["id"], "type": p["type"], "tirage": t,
                                        "response": txt}, ensure_ascii=False) + "\n")
                fait[0] += 1
                if fait[0] % 100 == 0:
                    print(f"  {fait[0]}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
            return
    with ThreadPoolExecutor(max_workers=args.parallele) as pool:
        list(pool.map(un, jobs))
    print(f"Terminé : {fait[0]} réponses écrites dans {sortie} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
