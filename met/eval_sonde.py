"""Tâche 2 : quels TYPES de prompts séparent les modèles malgré les déguisements ? (réponses des endpoints ref-)

Clé (practice_answers.csv) : S = ref-01/03/05/06/08 (normal, prompt système, persona, tatouage, température 1,0),
T1 = ref-02 (fp8), T3 = ref-04, T4 = ref-07, T2 = ref-09 (LoRA 0,7 / 1,5 / 0,3). 36 paires : 10 « même modèle »,
26 « différent » (toutes dans la même famille : c'est exactement la partie à 90 % du score).
Pour chaque prompt, deux mesures entre endpoints i et j (sur leurs R réponses) :
  - mmd  : MMD² de MET avec un noyau de Hamming sur les --L premiers caractères normalisés ;
  - cos  : 1 - k_ij / sqrt(k_ii k_jj) avec un noyau « réponse normalisée identique » (k_ii hors diagonale) :
           robuste à la température (qui étale sans déplacer).
Pour chaque type : distance = moyenne des rangs sur ses prompts ; AUC (différent > même) ; validation par moitiés
(prompts 1-3 contre 4-5) pour vérifier qu'un bon type le reste sur des prompts qu'on n'a pas regardés.
Enfin : MCC « même modèle » d'un regroupement à seuil (moyenne sur les types retenus), comme score_practice.py.

  python met/eval_sonde.py --reponses $OFFICIEL/model_equality/output/huang25/sonde_ref.jsonl
  python met/eval_sonde.py --fake
"""
import argparse
import itertools
import json
import math
import re
from collections import defaultdict

S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
MODELE = {**{e: "S" for e in S}, "ref-02": "T1", "ref-04": "T3", "ref-07": "T4", "ref-09": "T2"}
REFS = sorted(MODELE)
PAIRES = list(itertools.combinations(REFS, 2))


def norm(t):
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"^[\W_]+|[\W_]+$", "", t)


def hamming(a, b, L):
    a, b = a[:L].ljust(L), b[:L].ljust(L)
    return sum(x == y for x, y in zip(a, b)) / L


def mesures(ra, rb, L):
    """ra, rb : listes de réponses normalisées. Renvoie (mmd, cos)."""
    def k_moy(x, y, meme, f):
        v = [f(a, b) for i, a in enumerate(x) for j, b in enumerate(y) if not (meme and i == j)]
        return sum(v) / len(v) if v else 0.0
    h = lambda a, b: hamming(a, b, L)
    mmd = k_moy(ra, ra, True, h) + k_moy(rb, rb, True, h) - 2 * k_moy(ra, rb, False, h)
    e = lambda a, b: float(a[:L] == b[:L])
    kaa, kbb, kab = k_moy(ra, ra, True, e), k_moy(rb, rb, True, e), k_moy(ra, rb, False, e)
    cos = 1 - kab / math.sqrt(max(kaa, 1e-3) * max(kbb, 1e-3))
    return mmd, cos


def auc(dist, paires):
    meme = [dist[p] for p in paires if MODELE[p[0]] == MODELE[p[1]]]
    diff = [dist[p] for p in paires if MODELE[p[0]] != MODELE[p[1]]]
    return sum((d > m) + 0.5 * (d == m) for d in diff for m in meme) / (len(diff) * len(meme))


def rangs(d):
    ordre = sorted(d, key=d.get)
    return {p: i / (len(ordre) - 1) for i, p in enumerate(ordre)}


def mcc_seuil(dist):
    """Meilleure MCC « même modèle » d'un regroupement par composantes connexes à seuil (borne optimiste)."""
    best = (-1, None)
    for t in sorted(set(dist.values())):
        parent = {e: e for e in REFS}
        def f(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for (a, b), v in dist.items():
            if v <= t:
                parent[f(a)] = f(b)
        tp = fp = fn = tn = 0
        for a, b in PAIRES:
            pred = f(a) == f(b); vrai = MODELE[a] == MODELE[b]
            tp += pred and vrai; fp += pred and not vrai; fn += (not pred) and vrai; tn += (not pred) and not vrai
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1
        best = max(best, ((tp * tn - fp * fn) / den, t))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reponses")
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    R = defaultdict(list); typ = {}
    if args.fake:
        import random
        rnd = random.Random(0)
        for p in range(10):
            for e in REFS:
                for _ in range(4):
                    R[(f"p{p}", e)].append(norm(rnd.choice(["42", "43"]) if MODELE[e] == "S" else rnd.choice(["41", "44"])))
            typ[f"p{p}"] = "t" + str(p % 2)
    else:
        for l in open(args.reponses, encoding="utf-8"):
            r = json.loads(l); R[(r["id"], r["endpoint"])].append(norm(r["response"])); typ[r["id"]] = r["type"]
    prompts = sorted(typ)
    D = {m: {} for m in ("mmd", "cos")}
    for p in prompts:
        if not all(R[(p, e)] for e in REFS):
            continue
        for m in D:
            D[m][p] = {}
        for a, b in PAIRES:
            mmd, cos = mesures(R[(p, a)], R[(p, b)], args.L)
            D["mmd"][p][(a, b)] = mmd; D["cos"][p][(a, b)] = cos
    print(f"{len(D['mmd'])} prompts complets ; 36 paires (10 même modèle, 26 différentes)\n")
    print(f"{'type':14s} {'mesure':6s} {'AUC':>6s} {'AUC 1-3':>8s} {'AUC 4-5':>8s}   AUC par prompt")
    res = {}
    for t in sorted(set(typ.values())):
        ps = [p for p in prompts if typ[p] == t and p in D["mmd"]]
        for m in D:
            moy = lambda sel: {q: sum(rangs(D[m][p])[q] for p in sel) / len(sel) for q in PAIRES}
            a = auc(moy(ps), PAIRES); a1 = auc(moy(ps[:3]), PAIRES); a2 = auc(moy(ps[3:]), PAIRES) if ps[3:] else float("nan")
            res[(t, m)] = (a, a1, a2, moy(ps))
            print(f"{t:14s} {m:6s} {a:6.3f} {a1:8.3f} {a2:8.3f}   " + " ".join(f"{auc(D[m][p], PAIRES):.2f}" for p in ps))
    # combinaison : types dont l'AUC 1-3 >= 0,75 (choix sur la moitié 1-3), mesurée sur la moitié 4-5
    for m in D:
        bons = [t for (t, mm), v in res.items() if mm == m and v[1] >= 0.75]
        if not bons:
            continue
        ps45 = [p for p in prompts if typ[p] in bons and p in D[m] and p.endswith(("_4", "_5"))]
        ps_all = [p for p in prompts if typ[p] in bons and p in D[m]]
        if ps45:
            moy45 = {q: sum(rangs(D[m][p])[q] for p in ps45) / len(ps45) for q in PAIRES}
            moyall = {q: sum(rangs(D[m][p])[q] for p in ps_all) / len(ps_all) for q in PAIRES}
            print(f"\n{m} : types choisis sur les prompts 1-3 : {bons}")
            print(f"  validation sur les prompts 4-5 : AUC {auc(moy45, PAIRES):.3f} ; MCC (seuil optimal, optimiste) "
                  f"{mcc_seuil(moy45)[0]:.3f}")
            print(f"  tous les prompts de ces types : AUC {auc(moyall, PAIRES):.3f} ; MCC {mcc_seuil(moyall)[0]:.3f}")
            dm = moyall
            print("  distances moyennes (rang) : même modèle S " +
                  f"{sum(dm[p] for p in PAIRES if MODELE[p[0]] == MODELE[p[1]]) / 10:.3f} ; "
                  + " ; ".join(f"S-{x} {sum(dm[p] for p in PAIRES if {MODELE[p[0]], MODELE[p[1]]} == {'S', x}) / 5:.3f}"
                               for x in ("T1", "T2", "T3", "T4")))


if __name__ == "__main__":
    main()
