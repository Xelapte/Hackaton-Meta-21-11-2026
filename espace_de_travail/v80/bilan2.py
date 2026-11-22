"""v80 : bilan par endpoint : z poole (nouvelles reponses), z phase A (CSV de l'equipe, hors identite), z combine, T_hat."""
import csv, math, sys, collections
O = "/p/scratch/training2666/huang25/Epi84/model_equality/output/"
old = collections.defaultdict(lambda: [0.0, 0.0, 0])
for f in ["temp_ep_A.csv", "temp_ep_B.csv", "temp_ep_C.csv"]:
    for r in csv.DictReader(open(O + f)):
        if abs(float(r["T"]) - 0.7) < 1e-6 and r["id"] != "A01_identite":
            a = old[r["endpoint"]]; a[0] += float(r["ll"]) - float(r["E"]); a[1] += float(r["V"]); a[2] += int(r["n"])
for f in sys.argv[1:]:
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0, 0, 0])); per = collections.defaultdict(list)
    typ = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
    for r in csv.DictReader(open(f)):
        T = float(r["T"]); a = acc[r["endpoint"]][T]; d = float(r["ll"]) - float(r["E"]); v = float(r["V"])
        a[0] += d; a[1] += v; a[2] += int(r["n"]); a[3] += 1
        if abs(T - 0.7) < 1e-6:
            per[r["endpoint"]].append(d / math.sqrt(v)); t = typ[r["endpoint"]][r["id"][:8]]; t[0] += d; t[1] += v
    print(f"== {f.split('/')[-1]}")
    for ep in sorted(acc):
        a = acc[ep][0.7]; zn = a[0] / math.sqrt(a[1]); o = old[ep]; zo = o[0] / math.sqrt(o[1]) if o[1] else float("nan")
        zc = (a[0] + o[0]) / math.sqrt(a[1] + o[1])
        Ts = sorted(acc[ep]); zs = [acc[ep][T][0] / math.sqrt(acc[ep][T][1]) for T in Ts]; th = float("nan")
        for i in range(len(Ts) - 1):
            if zs[i] * zs[i + 1] <= 0:
                th = Ts[i] + (Ts[i + 1] - Ts[i]) * zs[i] / (zs[i] - zs[i + 1])
        p = per[ep]; m = sum(p) / len(p); sd = (sum((x - m) ** 2 for x in p) / max(1, len(p) - 1)) ** 0.5
        print(f"{ep} : nouvelles {a[3]:3d} rep {a[2]:6d} tok  z_new {zn:+6.2f} | phaseA z {zo:+5.2f} | combine {zc:+6.2f} | T_hat {th:.3f} | z/rep moy {m:+.3f} (t={m / (sd / math.sqrt(len(p))):+.2f}, sd {sd:.2f})")
        print("      par type : " + "  ".join(f"{k}:{v[0] / math.sqrt(v[1]):+.1f}" for k, v in sorted(typ[ep].items())))
