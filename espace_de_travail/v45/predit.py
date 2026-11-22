#!/usr/bin/env python
"""v45 - Prediction du modele du leaderboard pour des reglages (A, B, C) donnes. Usage : python predit.py "13,13,0" "16,4,0" ..."""
import sys, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
c = torch.load(V + "cache.pt"); r = torch.load(V + "cache_record.pt"); y = c["y"]
Ze = r["record ext_wr1"].float(); Za = r["record ext_wr1 | copies"].float()
A, Bq, C = [0, 38], [6, 7, 8], [29, 30, 31, 33]
def stats(spec):
    out = []
    for Z in (Ze, Za):
        z = Z.clone()
        for cl, b in spec.items(): z[:, cl] -= b
        L = F.cross_entropy(z, y, reduction="none"); out += [L.mean().item(), L.std().item()]
    return out
FITS = {"sans cout retain": [(1.25, 9.0, 0.485, 0.5, -0.5), (1.25, 6.5, 0.367 * 1.2 / 1.2, 0.5, -0.5), (1.25, 5.5, 0.266, 0.5, -0.5)],   # (MU, SIGMA, c/(1+K0), dM, dS)
        "std gold 4.25-5": [(1.5, 4.25, 0.279 / 1.2, 0.5, 0.5), (1.5, 4.5, 0.290 / 1.2, 0.5, 0.5), (1.5, 5.0, 0.313 / 1.2, 0.5, 0.5)]}
FITS["sans cout retain"] = [(1.25, 9.0, 0.485 / 1.2, 0.5, -0.5), (1.25, 6.5, 0.367 / 1.2, 0.5, -0.5), (1.25, 5.5, 0.266, 0.5, -0.5)]
def pred(par, st):
    MU, SG, cst, DM, DS = par; ms = 0.0
    for (m, s, mu, sg) in ((st[0], st[1], MU, SG), (st[2], st[3], MU + DM, SG + DS)): ms += 0.5 * (1 - min(1.0, (abs(m - mu) + abs(s - sg)) / (mu + sg)))
    return cst * ms
for a in sys.argv[1:]:
    v = [float(x) for x in a.split(",")]; sp = {**{k: v[0] for k in A}, **{k: v[1] for k in Bq}, **({k: v[2] for k in C} if len(v) > 2 and v[2] else {})}
    if len(v) > 3:                                                      # A separe : classe 0 = v[0], classe 38 = v[3]
        sp[38] = v[3]
    st = stats(sp)
    print(f"{a:14} m/s exactes {st[0]:.2f}/{st[1]:.2f} copies {st[2]:.2f}/{st[3]:.2f} | " + " | ".join(f"{k} : " + " ".join(f"{pred(p, st):.3f}" for p in ps) for k, ps in FITS.items()))
