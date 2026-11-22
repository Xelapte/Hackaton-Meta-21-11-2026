"""Tâche 2 : analyse des sorties de calibration.py, par endpoint.

Par réponse : calib = (lp + H) / n (≈ 0 si l'endpoint tire ses tokens de p_base à température 1 ; > 0 si
température < 1 ; < 0 si les tokens viennent d'une autre loi, ex. LoRA), et top1 = part des tokens égaux à l'argmax.
Par endpoint : moyenne et erreur type sur ses réponses ; sur les ref-, rappel de la clé.
Comparaison appariée : pour chaque (prompt, tirage), écart de calib à la médiane des autres endpoints de la
famille sur ce même prompt (enlève la difficulté propre du prompt) ; z = moyenne / erreur type.

  python met/analyse_calibration.py --calib out/calib_ref_v2.csv --key practice_answers.csv
"""
import argparse
import csv
import statistics as st
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--exclure", default="A01_identite", help="ids de prompts à ignorer (virgules)")
    args = ap.parse_args()
    excl = set(args.exclure.split(","))
    cle = {}
    if args.key:
        for r in csv.reader(open(args.key, encoding="utf-8-sig")):
            if r and r[0] != "endpoint":
                cle[r[0]] = f"{r[2]:3s} {r[3][:22]}"
    rep = defaultdict(dict)                       # endpoint -> (id, repeat) -> (calib, top1)
    for f in args.calib:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["id"] in excl or int(r["n"]) < 5:
                continue
            n = int(r["n"])
            rep[r["endpoint"]][(r["id"], r["repeat"])] = ((float(r["lp"]) + float(r["H"])) / n, int(r["top1"]) / n,
                                                          float(r["lp"]) / n)
    eps = sorted(rep)
    par_prompt = defaultdict(dict)
    for e in eps:
        for k, v in rep[e].items():
            par_prompt[k[0]].setdefault(e, []).append(v)

    def stat(xs):
        m = st.mean(xs); et = st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("inf")
        return m, et

    print(f"{'endpoint':8s} {'clé':27s} {'n':>3s} {'calib':>15s} {'top1':>7s} {'lp/tok':>7s} | écart apparié calib (z)   top1 (z)")
    lignes = []
    for e in eps:
        c = [v[0] for v in rep[e].values()]; t = [v[1] for v in rep[e].values()]; l = [v[2] for v in rep[e].values()]
        dc, dt = [], []
        for p, d in par_prompt.items():
            if e not in d:
                continue
            autres_c = [x[0] for o, vs in d.items() if o != e for x in vs]
            autres_t = [x[1] for o, vs in d.items() if o != e for x in vs]
            if len(autres_c) < 3:
                continue
            for x in d[e]:
                dc.append(x[0] - st.median(autres_c)); dt.append(x[1] - st.median(autres_t))
        mc, ec = stat(c); (mdc, edc), (mdt, edt) = stat(dc), stat(dt)
        lignes.append((e, mdc / edc))
        print(f"{e:8s} {cle.get(e, ''):27s} {len(c):3d} {mc:+.3f} ± {ec:.3f} {st.mean(t):7.3f} {st.mean(l):+7.3f} | "
              f"{mdc:+.3f} ({mdc / edc:+6.2f})   {mdt:+.3f} ({mdt / edt:+6.2f})")


if __name__ == "__main__":
    main()
