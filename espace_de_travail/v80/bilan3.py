"""v80 : lecture des sorties de score3.py. chi2 des comptes agreges (tokens d'attendu >= --emin, choisis sur l'attendu poole de tous
les endpoints du fichier, donc sans regarder les observations) + z de vraisemblance."""
import sys, math, torch
emin = 15.0
for f in sys.argv[1:]:
    res = torch.load(f); eps = sorted(res); ids = sorted({i for e in eps for i in res[e]["par_id"]})
    print("==", f.split("/")[-1], "| ids :", ids)
    for grp in [ids] + ([[i] for i in ids] if len(ids) > 1 else []):
        tot = {}
        for e in eps:
            O = sum(res[e]["par_id"][i]["O"] for i in grp if i in res[e]["par_id"]); E = sum(res[e]["par_id"][i]["E"] for i in grp if i in res[e]["par_id"])
            W = sum(res[e]["par_id"][i]["W"] for i in grp if i in res[e]["par_id"])
            z = [x for i in grp if i in res[e]["par_id"] for x in res[e]["par_id"][i]["z"]]
            tot[e] = (O, E, W, z)
        Ep = sum(t[1] for t in tot.values()) / len(tot)
        S0 = (Ep >= emin)
        line = []
        for e in eps:
            O, E, W, z = tot[e]
            if not z: continue
            S = (S0 & (W >= 3.0)).nonzero().flatten(); x = ((O[S] - E[S]) / W[S].sqrt())
            chi = float((x * x).sum()); k = max(1, len(S))
            zl = sum(a - b for a, b, _, _ in z) / math.sqrt(sum(v for _, _, v, _ in z))
            line.append(f"{e[-5:]}: chi2 {chi:6.1f}/{k} (z={(chi - k) / math.sqrt(2 * k):+5.1f}) zll {zl:+5.1f} n={sum(n for *_, n in z)}")
        print(("TOUT " if len(grp) > 1 else grp[0][:10].ljust(10)) + " | " + "  ||  ".join(line))
