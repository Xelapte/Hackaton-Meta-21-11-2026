"""Tâche 2 : quel prompt révèle quelle variante ? Analyse « un contre le centre » sur les endpoints ref-.

La MCC punit surtout deux erreurs : fusionner une variante (fp8, LoRA) avec le modèle de base, ou séparer un
déguisement du modèle de base. Confondre deux variantes entre elles coûte peu. La question utile est donc, pour
chaque endpoint : est-il le modèle de base (déguisé ou non) ?
Pour chaque prompt : centroïde TF-IDF (n-grammes de caractères 3–5) des réponses de chaque endpoint ;
distance au centre S d'un endpoint = moyenne des distances cosinus aux membres de S (hors lui-même).
Séparabilité d'une variante V : part des membres de S plus proches du centre que V (1 = V clairement à l'écart).
Agrégé par type de prompt et sur tous les prompts.

  python met/analyse_prompts.py --reponses f1.jsonl f2.jsonl ...
  python met/analyse_prompts.py --fake
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict

S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
NOM_S = {"ref-01": "normal", "ref-03": "syst", "ref-05": "persona", "ref-06": "tatouage", "ref-08": "temp1"}
V = {"ref-02": "fp8", "ref-09": "LoRA0.3", "ref-04": "LoRA0.7", "ref-07": "LoRA1.5"}
REFS = S + list(V)


def ngrammes(t):
    t = re.sub(r"\s+", " ", t.lower())
    return Counter(t[i:i + n] for n in (3, 4, 5) for i in range(len(t) - n + 1))


def centroides(rep):
    """rep : {endpoint: [textes]} -> {endpoint: vecteur TF-IDF moyen normalisé (dict)}."""
    docs = {e: [ngrammes(t) for t in ts] for e, ts in rep.items()}
    df = Counter(g for ds in docs.values() for d in ds for g in d)
    n = sum(len(ds) for ds in docs.values())
    idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
    out = {}
    for e, ds in docs.items():
        acc = Counter()
        for d in ds:
            v = {g: c * idf[g] for g, c in d.items()}; no = math.sqrt(sum(x * x for x in v.values())) or 1
            for g, x in v.items():
                acc[g] += x / no / len(ds)
        no = math.sqrt(sum(x * x for x in acc.values())) or 1
        out[e] = {g: x / no for g, x in acc.items()}
    return out


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(g, 0.0) for g, x in a.items())


def analyse_prompt(rep):
    c = centroides(rep)
    d = {(a, b): 1 - cos(c[a], c[b]) for a in REFS for b in REFS if a != b}
    centre = {e: sum(d[(e, s)] for s in S if s != e) / (len(S) - (e in S)) for e in REFS}
    sep = {v: sum(centre[s] < centre[v] for s in S) / len(S) for v in V}
    return centre, sep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reponses", nargs="+")
    ap.add_argument("--min_tirages", type=int, default=2)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    R = defaultdict(lambda: defaultdict(list)); typ = {}
    if args.fake:
        import random
        rnd = random.Random(0)
        for p in range(4):
            typ[f"p{p}"] = "t"
            for e in REFS:
                for _ in range(3):
                    R[f"p{p}"][e].append(" ".join(rnd.choice(["le chat", "un chien", "la pluie"] if e in S else ["zorg", "blip"]) for _ in range(8)))
    else:
        for f in args.reponses:
            src = f.split("/")[-1].split(".")[0][:10]
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if not r["endpoint"].startswith("ref-"):
                    continue
                pid = f"{src}:{r['id']}"; R[pid][r["endpoint"]].append(r["response"] or "")
                typ[pid] = r.get("type", r["id"])
    res = {}
    print(f"{'prompt':38s} {'tir':>3s}  sép. fp8 L0.3 L0.7 L1.5 | distance au centre : S (min–max)  / variantes")
    for p in sorted(R):
        if not all(len(R[p][e]) >= args.min_tirages for e in REFS):
            continue
        centre, sep = analyse_prompt(R[p]); res[p] = (centre, sep)
        smin, smax = min(centre[s] for s in S), max(centre[s] for s in S)
        pire = max(S, key=lambda s: centre[s])
        print(f"{p[:38]:38s} {min(len(R[p][e]) for e in REFS):3d}  " + " ".join(f"{sep[v]:4.1f}" for v in V)
              + f" | S {smin:.3f}–{smax:.3f} (plus loin : {NOM_S[pire]}) / "
              + " ".join(f"{V[v]} {centre[v]:.3f}" for v in V))
    if not res:
        print("pas assez de données"); return
    print("\nPar type de prompt (moyenne des séparabilités ; 1 = toutes les variantes plus loin que tout S) :")
    par_type = defaultdict(list)
    for p, (c, s) in res.items():
        par_type[typ[p]].append(s)
    for t, ss in sorted(par_type.items(), key=lambda kv: -sum(sum(s.values()) for s in kv[1]) / len(kv[1])):
        moy = {v: sum(s[v] for s in ss) / len(ss) for v in V}
        print(f"  {t:14s} ({len(ss):2d} prompts) : " + " ".join(f"{V[v]} {moy[v]:.2f}" for v in V)
              + f"   moyenne {sum(moy.values()) / 4:.2f}")
    # agrégat sur tous les prompts : distance au centre moyenne (rangs par prompt)
    rang = defaultdict(float)
    for p, (c, s) in res.items():
        ordre = sorted(REFS, key=lambda e: c[e])
        for i, e in enumerate(ordre):
            rang[e] += i / len(res)
    print("\nRang moyen de la distance au centre S, tous prompts (0 = le plus proche ; 8 = le plus loin) :")
    for e in sorted(REFS, key=lambda e: rang[e]):
        print(f"  {e} {NOM_S.get(e, V.get(e)):9s} {rang[e]:.2f}")


if __name__ == "__main__":
    main()
