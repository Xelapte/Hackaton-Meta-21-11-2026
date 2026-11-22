"""Tâche 2 : analyse des sorties de temperature.py, par endpoint.

  T_hat : température qui maximise la vraisemblance totale des réponses (interpolation parabolique sur la grille) ;
  Z(T)  : somme sur les réponses de (ll_T - E_T) / sqrt(somme de V_T) (≈ N(0, 1) si l'endpoint tire de p_T).
On affiche Z à T0 = médiane des T_hat de la famille (hors endpoints de T_hat > --t_temp, déguisement température)
et Z à T_hat (qualité d'ajustement à la meilleure température).

  python met/analyse_temperature.py --temp out/temp_ref_equipe.csv --key practice_answers.csv
"""
import argparse
import csv
import math
import statistics as st
from collections import defaultdict


def charger(fichiers, exclure):
    D = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0]))   # endpoint -> T -> [ll, E, V, n]
    for f in fichiers:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["id"] in exclure or int(r["n"]) < 5:
                continue
            a = D[r["endpoint"]][float(r["T"])]
            a[0] += float(r["ll"]); a[1] += float(r["E"]); a[2] += float(r["V"]); a[3] += int(r["n"])
    return D


def t_hat(d):
    Ts = sorted(d); ll = [d[T][0] for T in Ts]; i = max(range(len(Ts)), key=ll.__getitem__)
    if 0 < i < len(Ts) - 1:
        y0, y1, y2 = ll[i - 1], ll[i], ll[i + 1]; den = y0 - 2 * y1 + y2
        if den < 0:
            return Ts[i] + 0.5 * (y0 - y2) / den * (Ts[i + 1] - Ts[i])
    return Ts[i]


def Z(d, T):
    T = min(d, key=lambda t: abs(t - T))
    ll, E, V, n = d[T]
    return (ll - E) / math.sqrt(V)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temp", nargs="+", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--exclure", default="A01_identite")
    ap.add_argument("--t_temp", type=float, default=0.95)
    args = ap.parse_args()
    cle = {}
    if args.key:
        for r in csv.reader(open(args.key, encoding="utf-8-sig")):
            if r and r[0] != "endpoint":
                cle[r[0]] = f"{r[2]:3s} {r[3][:22]}"
    D = charger(args.temp, set(args.exclure.split(",")))
    th = {e: t_hat(D[e]) for e in D}
    T0 = st.median([t for t in th.values() if t <= args.t_temp])
    print(f"T0 (médiane des T_hat <= {args.t_temp}) = {T0:.3f}")
    print(f"{'endpoint':8s} {'clé':27s} {'T_hat':>6s} {'Z(T0)':>7s} {'Z(T_hat)':>8s} {'ll/tok(T0)':>10s}")
    for e in sorted(D, key=lambda e: Z(D[e], T0)):
        T = min(D[e], key=lambda t: abs(t - T0))
        print(f"{e:8s} {cle.get(e, ''):27s} {th[e]:6.3f} {Z(D[e], T0):+7.2f} {Z(D[e], th[e]):+8.2f} {D[e][T][0] / D[e][T][3]:+10.3f}")


if __name__ == "__main__":
    main()
