"""Tâche 2 (Model Equality) : regrouper les endpoints en familles et en modèles.

Lit les réponses collectées par les scripts officiels (answers.jsonl pour ep-XX,
reference_answers.jsonl pour ref-XX), calcule une similarité entre chaque paire
d'endpoints, puis fait un regroupement hiérarchique à deux seuils (famille, modèle).

Usage, dans le venv officiel model_equality (CPU, sur le nœud de login : c'est léger) :
  # Régler les seuils sur les endpoints de référence (réponses connues, sans quota)
  python met/analyse.py --answers <officiel>/model_equality/reference_answers.jsonl \
      --prefix ref- --key <officiel>/model_equality/practice_answers.csv
  # Produire la soumission sur les vrais endpoints avec les seuils retenus
  python met/analyse.py --answers <officiel>/model_equality/answers.jsonl --prefix ep- \
      --family_thr 0.35 --model_thr 0.10 --out submission.csv

Similarité entre deux endpoints = moyenne, sur les prompts communs, de la similarité
cosinus TF-IDF (n-grammes de caractères) entre leurs réponses. C'est une baseline :
à améliorer (voir docs/TACHE_2_model_equality.md).
"""
import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_answers(paths, prefix):
    """{endpoint: {prompt_id: [réponses]}} pour les endpoints qui commencent par prefix."""
    data = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if str(r["endpoint"]).startswith(prefix):
                    data[r["endpoint"]][str(r["id"])].append(r["response"] or "")
    return {e: dict(v) for e, v in sorted(data.items())}


def similarity_matrix(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    eps = list(data)
    texts, index = [], {}
    for e in eps:
        for pid, resps in data[e].items():
            for k, t in enumerate(resps):
                index[(e, pid, k)] = len(texts)
                texts.append(t)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                          min_df=1).fit(texts)
    X = vec.transform(texts)                       # lignes normalisées L2
    S = np.eye(len(eps))
    n_common = np.zeros((len(eps), len(eps)), dtype=int)
    for i, j in itertools.combinations(range(len(eps)), 2):
        a, b = eps[i], eps[j]
        common = sorted(set(data[a]) & set(data[b]))
        sims = []
        for pid in common:
            ia = [index[(a, pid, k)] for k in range(len(data[a][pid]))]
            ib = [index[(b, pid, k)] for k in range(len(data[b][pid]))]
            sims.append(float((X[ia] @ X[ib].T).mean()))
        S[i, j] = S[j, i] = np.mean(sims) if sims else 0.0
        n_common[i, j] = n_common[j, i] = len(common)
    return eps, S, n_common


def cluster(D, thr):
    """Regroupement hiérarchique (lien moyen) ; renvoie un label entier par endpoint."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(D) == 1:
        return np.array([1])
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=thr, criterion="distance")


def letters(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def assign(eps, S, family_thr, model_thr):
    D = np.clip(1.0 - S, 0.0, None)
    fam = cluster(D, family_thr)
    rows = []
    for f_i, f in enumerate(sorted(set(fam))):
        idx = np.where(fam == f)[0]
        sub = cluster(D[np.ix_(idx, idx)], model_thr) if len(idx) > 1 else np.array([1])
        flab = letters(f_i)
        for k, m in zip(idx, sub):
            rows.append((eps[k], flab, f"{flab}{m}"))
    return sorted(rows)


def mcc(pairs_true, pairs_pred):
    tp = sum(t and p for t, p in zip(pairs_true, pairs_pred))
    tn = sum((not t) and (not p) for t, p in zip(pairs_true, pairs_pred))
    fp = sum((not t) and p for t, p in zip(pairs_true, pairs_pred))
    fn = sum(t and (not p) for t, p in zip(pairs_true, pairs_pred))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / den) if den else 0.0


def score(rows, key):
    """Score du serveur : 0,1 × MCC famille (toutes paires) + 0,9 × MCC modèle
    (paires d'une même famille, d'après la vérité)."""
    pred = {e: (f, m) for e, f, m in rows}
    eps = sorted(set(pred) & set(key))
    fam_t, fam_p, mod_t, mod_p = [], [], [], []
    for a, b in itertools.combinations(eps, 2):
        fam_t.append(key[a][0] == key[b][0])
        fam_p.append(pred[a][0] == pred[b][0])
        if key[a][0] == key[b][0]:
            mod_t.append(key[a] == key[b])
            mod_p.append(pred[a] == pred[b])
    m_fam, m_mod = mcc(fam_t, fam_p), mcc(mod_t, mod_p)
    return 0.1 * m_fam + 0.9 * m_mod, m_fam, m_mod


def read_key(path):
    key = {}
    with open(path) as fh:
        for r in csv.reader(fh):
            if r and not r[0].startswith("endpoint"):
                key[r[0]] = (r[1], r[2])
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--prefix", default="ep-", help="ep- (soumission) ou ref- (réglage)")
    ap.add_argument("--key", help="practice_answers.csv : active la recherche des seuils")
    ap.add_argument("--family_thr", type=float, default=0.35)
    ap.add_argument("--model_thr", type=float, default=0.10)
    ap.add_argument("--out", default=None, help="CSV endpoint,family,model")
    ap.add_argument("--show", action="store_true", help="affiche la matrice de similarité")
    args = ap.parse_args()

    data = load_answers(args.answers, args.prefix)
    if len(data) < 2:
        sys.exit(f"Moins de 2 endpoints '{args.prefix}' dans {args.answers}")
    eps, S, n_common = similarity_matrix(data)
    print(f"{len(eps)} endpoints, prompts communs par paire : min {n_common[n_common > 0].min()}"
          f", médiane {int(np.median(n_common[n_common > 0]))}")
    if args.show:
        print("      " + " ".join(e[-2:].rjust(5) for e in eps))
        for e, row in zip(eps, S):
            print(e[-6:].rjust(6) + " ".join(f"{v:5.2f}" for v in row))

    ft, mt = args.family_thr, args.model_thr
    if args.key:
        key = read_key(args.key)
        best = (-1, ft, mt)
        for f_thr in np.linspace(0.05, 0.95, 19):
            for m_thr in np.linspace(0.0, f_thr, 20):
                s = score(assign(eps, S, f_thr, m_thr), key)[0]
                if s >= best[0] - 1e-9:      # à égalité : seuils les plus larges
                    best = (s, f_thr, m_thr)
        _, ft, mt = best
        s, sf, sm = score(assign(eps, S, ft, mt), key)
        print(f"Meilleurs seuils sur la clé : family_thr={ft:.3f} model_thr={mt:.3f} "
              f"-> score {s:.3f} (MCC famille {sf:.3f}, modèle {sm:.3f})")
        print("Attention : 9 endpoints de référence = peu de paires ; seuils indicatifs.")

    rows = assign(eps, S, ft, mt)
    for r in rows:
        print(",".join(r))
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["endpoint", "family", "model"])
            w.writerows(rows)
        print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
