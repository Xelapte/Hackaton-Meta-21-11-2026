"""Tâche 2 : collecte des réponses, avec garde-fous (le budget des ep- est de 100 réponses par endpoint, définitif).

Garde-fous :
  - --plafond : nombre MAXIMAL de réponses par endpoint dans le fichier de sortie (toutes phases confondues) ;
    le script ne demande jamais au-delà, même relancé ;
  - pour les ep- : lecture du budget sur le serveur (GET /models, gratuit) ; refus si déjà utilisé + prévu
    dépasse --plafond_serveur ; et --confirmer obligatoire (sinon : simulation à blanc) ;
  - --a_blanc : affiche le plan (requêtes par endpoint, total, durée estimée) sans rien envoyer ;
  - ordre des requêtes : tirage, puis prompt, puis endpoint : si la collecte est coupée, tous les endpoints
    ont à peu près autant de réponses ;
  - reprise : les (endpoint, id, tirage) déjà présents ne sont pas redemandés ;
  - une requête invalide (4xx hors 429) n'est pas comptée par le serveur : elle est journalisée, pas réessayée.
La clé est lue dans TEAM_API_KEY (jamais affichée).

Répétition générale (références, gratuites) :
  python met/collecte.py --prompts met/prompts_phaseA.csv --endpoints ref --tirages 3 --plafond 30 \\
      --sortie $OFFICIEL/model_equality/output/huang25/repetition_ref.jsonl
Vrais endpoints (décision d'équipe) : d'abord à blanc, puis avec --confirmer :
  python met/collecte.py --prompts met/prompts_phaseA.csv --endpoints ep --tirages 3 --plafond 30 \\
      --sortie $OFFICIEL/model_equality/output/collecte_ep.jsonl [--confirmer]
Test sans réseau : python met/collecte.py --prompts met/prompts_phaseA.csv --endpoints ref --fake
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://34.147.229.55/33-model-equality/v1"


def endpoints(spec):
    if spec == "ep":
        return [f"ep-{i:02d}" for i in range(1, 34)]
    if spec == "ref":
        return [f"ref-{i:02d}" for i in range(1, 10)]
    return spec.split(",")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--endpoints", required=True, help="ep | ref | liste séparée par des virgules")
    ap.add_argument("--tirages", type=int, required=True)
    ap.add_argument("--plafond", type=int, required=True, help="réponses max par endpoint dans --sortie")
    ap.add_argument("--plafond_serveur", type=int, default=80, help="ep- : utilisé + prévu ne doit pas dépasser")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--parallele", type=int, default=4)
    ap.add_argument("--secondes_par_requete", type=float, default=12.0, help="pour l'estimation de durée")
    ap.add_argument("--essais", type=int, default=8, help="tentatives par requête (ep- : 1 recommandé, voir --aide_budget)")
    ap.add_argument("--delai", type=float, default=300, help="délai d'attente d'une réponse (s)")
    ap.add_argument("--a_blanc", action="store_true")
    ap.add_argument("--confirmer", action="store_true")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    eps = endpoints(args.endpoints)
    vrais = [e for e in eps if e.startswith("ep-")]
    prompts = list(csv.DictReader(open(args.prompts, encoding="utf-8")))
    assert all(int(p["max_tokens"]) <= 256 for p in prompts), "max_tokens > 256 refusé par le serveur"
    ids = [p["id"] for p in prompts]
    assert len(set(ids)) == len(ids), "identifiants de prompts en double"
    sortie = Path(args.sortie); sortie.parent.mkdir(parents=True, exist_ok=True)
    deja, par_ep = set(), Counter()
    if sortie.exists():
        for l in open(sortie, encoding="utf-8"):
            r = json.loads(l); deja.add((r["endpoint"], r["id"], r["tirage"])); par_ep[r["endpoint"]] += 1
    jobs, prevu = [], Counter()
    n_tir = {p["id"]: int(p.get("tirages") or args.tirages) for p in prompts}   # colonne « tirages » facultative
    for t in range(max(n_tir.values())):
        for p in prompts:
            if t >= n_tir[p["id"]]:
                continue
            for e in eps:
                if (e, p["id"], t) in deja or par_ep[e] + prevu[e] >= args.plafond:
                    continue
                jobs.append((e, p, t)); prevu[e] += 1
    duree = len(jobs) * args.secondes_par_requete / max(1, args.parallele) / 60
    print(f"Plan : {len(prompts)} prompts, {sum(n_tir.values())} réponses par endpoint au total, {len(eps)} endpoints, plafond {args.plafond}/endpoint")
    print(f"  déjà dans le fichier : {sum(par_ep.values())} ; à demander : {len(jobs)} "
          f"(par endpoint : min {min(prevu[e] for e in eps)}, max {max(prevu[e] for e in eps)}) ; "
          f"durée estimée ~{duree:.0f} min à {args.parallele} en parallèle", flush=True)
    if args.fake:
        import random
        rnd = random.Random(0)
        with open(sortie, "a", encoding="utf-8") as f:
            for e, p, t in jobs:
                f.write(json.dumps({"endpoint": e, "id": p["id"], "type": p["type"], "tirage": t,
                                    "max_tokens": int(p["max_tokens"]), "response": rnd.choice(["a b c", "a b d"]),
                                    "heure": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n")
        print(f"(faux) écrit : {sortie}"); return
    import requests
    cle = os.environ.get("TEAM_API_KEY", "")
    if not cle:
        sys.exit("TEAM_API_KEY n'est pas définie.")
    s = requests.Session(); s.headers["X-API-Key"] = cle
    if vrais:
        budget = {m["id"]: m["budget"] for m in s.get(f"{BASE}/models", timeout=60).json()["data"]}
        trop = [(e, budget[e]["used"], prevu[e]) for e in vrais if budget[e]["used"] + prevu[e] > args.plafond_serveur]
        print(f"  budget serveur : utilisé min {min(budget[e]['used'] for e in vrais)}, max {max(budget[e]['used'] for e in vrais)} "
              f"(limite {next(iter(budget.values()))['limit']}) ; plafond serveur fixé à {args.plafond_serveur}", flush=True)
        if trop:
            sys.exit(f"REFUSÉ : utilisé + prévu > {args.plafond_serveur} pour {trop[:5]}…")
        if not args.confirmer:
            print("Simulation à blanc (ep-) : ajoutez --confirmer pour envoyer. Rien n'a été envoyé."); return
    if args.a_blanc:
        print("À blanc : rien n'a été envoyé."); return
    verrou = threading.Lock(); fait = [0]; erreurs = Counter(); t0 = time.time()

    causes = Counter()

    def un(job):
        e, p, t = job
        essai = 0
        while essai < args.essais:
            essai += 1
            try:
                r = s.post(f"{BASE}/chat/completions", timeout=args.delai,
                           json={"model": e, "max_tokens": int(p["max_tokens"]),
                                 "messages": [{"role": "user", "content": p["prompt"]}]})
            except requests.RequestException as exc:
                with verrou:
                    causes[type(exc).__name__] += 1
                time.sleep(15); continue
            if r.status_code == 429:                      # refusé, non exécuté, non décompté : on attend sans compter l'essai
                with verrou:
                    causes["429"] += 1
                essai -= 1
                time.sleep(int(r.headers.get("Retry-After", 15)) + 5); continue
            if r.status_code >= 500:
                with verrou:
                    causes[f"HTTP {r.status_code}: {r.text[:60]}"] += 1
                time.sleep(15); continue
            if r.status_code != 200:
                with verrou:
                    erreurs[r.status_code] += 1
                    print(f"  {e} {p['id']} t{t} : HTTP {r.status_code} {r.text[:150]}", flush=True)
                return
            txt = r.json()["choices"][0]["message"]["content"]
            with verrou:
                with open(sortie, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"endpoint": e, "id": p["id"], "type": p["type"], "tirage": t,
                                        "max_tokens": int(p["max_tokens"]), "response": txt,
                                        "heure": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n")
                fait[0] += 1
                if fait[0] % 50 == 0:
                    v = fait[0] / (time.time() - t0) * 60
                    print(f"  {fait[0]}/{len(jobs)} ({v:.1f}/min, reste ~{(len(jobs) - fait[0]) / max(v, 1e-6):.0f} min)", flush=True)
            return
        with verrou:
            erreurs["abandon"] += 1
    with ThreadPoolExecutor(max_workers=args.parallele) as pool:
        list(pool.map(un, jobs))
    print(f"Terminé : {fait[0]} réponses en {(time.time() - t0) / 60:.0f} min ; erreurs {dict(erreurs)} ; "
          f"causes des nouvelles tentatives : {dict(causes)}", flush=True)


if __name__ == "__main__":
    main()
