"""Tâche 2 : analyser les statistiques token par token de met/logprobs_local.py (CPU, léger).

Par endpoint, et pour les tokens au-delà de --pos_min (loin du début, là où le prompt caché pèse moins) :
  T*       température qui maximise la vraisemblance des tokens sous le modèle local
  lp(T*)   log-vraisemblance moyenne par token à T* (plus haut = colle mieux au modèle local)
  top1     part des tokens qui sont l'argmax du modèle local
  KS       écart de l'histogramme des PIT (à T*) à l'uniforme (Kolmogorov-Smirnov ; petit = bon modèle)
  LLR fp8  log p(base) - log p(fp8) par token à T* (si --fp8 est donné)

  python met/analyse_logprobs.py --npz results/logprobs_ref.npz --fp8 results/logprobs_ref_fp8.npz \\
      --key <officiel>/model_equality/practice_answers.csv
"""
import argparse
import csv
import os
import sys

import numpy as np


def lire_cle(chemin):
    cle = {}
    if chemin:
        with open(chemin, newline="", encoding="utf-8-sig") as fh:
            for r in csv.reader(fh):
                if r and not r[0].startswith("endpoint"):
                    cle[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return cle


def ks_uniforme(u):
    if len(u) == 0:
        return float("nan")
    u = np.sort(u)
    n = len(u)
    return float(max((np.arange(1, n + 1) / n - u).max(), (u - np.arange(n) / n).max()))


def masque_endpoint(D, e, pos_min, pos_max):
    """Indices des tokens de l'endpoint e, entre pos_min et pos_max."""
    idx = []
    for i in np.where(D["endpoint"] == e)[0]:
        idx.append(np.arange(D["debut"][i], D["fin"][i]))
    idx = np.concatenate(idx) if idx else np.array([], dtype=int)
    p = D["pos"][idx]
    return idx[(p >= pos_min) & (p < pos_max)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fp8", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--pos_min", type=int, nargs="+", default=[0, 20, 50])
    ap.add_argument("--pos_max", type=int, default=10 ** 9)
    ap.add_argument("--ent_max", type=float, default=None, help="ne garder que les tokens d'entropie < ent_max")
    args = ap.parse_args()

    D = dict(np.load(args.npz))
    F = dict(np.load(args.fp8)) if args.fp8 else None
    cle = lire_cle(args.key)
    temps = D["temps"]
    eps = sorted(set(D["endpoint"]))
    lab = lambda e: f"{e} {cle[e][1]:>3s} {cle[e][2][:20]:20s}" if e in cle else f"{e:28s}"

    for pm in args.pos_min:
        print(f"\n=== tokens en position >= {pm}" + (f", entropie < {args.ent_max}" if args.ent_max else ""))
        print(f"{'':28s} {'ntok':>6s} {'T*':>5s} {'lp(T*)':>7s} {'top1':>6s} {'KS':>6s} {'ent':>5s}"
              + (f" {'LLRfp8':>8s} {'z':>6s}" if F else ""))
        for e in eps:
            idx = masque_endpoint(D, e, pm, args.pos_max)
            if args.ent_max is not None:
                idx = idx[D["ent"][idx] < args.ent_max]
            if len(idx) == 0:
                print(lab(e), "aucun token"); continue
            lp = D["lp"][idx]                                  # (n, nT)
            k = int(lp.mean(0).argmax())
            ligne = (f"{lab(e)} {len(idx):6d} {temps[k]:5.2f} {lp[:, k].mean():7.3f} "
                     f"{(D['rang'][idx] == 0).mean():6.3f} {ks_uniforme(D['pit'][idx, k]):6.3f} "
                     f"{D['ent'][idx].mean():5.2f}")
            if F:
                d = D["lp"][idx, k] - F["lp"][idx, k]
                z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                ligne += f" {d.mean():+8.4f} {z:+6.2f}"
            print(ligne)


if __name__ == "__main__":
    main()
