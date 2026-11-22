#!/usr/bin/env python
"""v45 - Modele du leaderboard, version 2 (vectorisee) : note = c x moyenne(exactes, copies) de [1 - (|m - MU| + |s - SIGMA|) / (MU + SIGMA)].
On ajuste (MU, SIGMA) du gold + decalage pour les copies sur toutes nos notes, puis on classe des reglages candidats.
Usage : python fit2.py [nom:spec_python:note ...]   (points supplementaires)"""
import sys, os, itertools, json, ast, numpy as np, torch, torch.nn.functional as F
TL = float(os.environ.get("TL", "1"))
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
c = torch.load(V + "cache.pt"); r = torch.load(V + "cache_record.pt"); y = c["y"]
Ze = r["record ext_wr1"].float(); Za = r["record ext_wr1 | copies"].float()
A, Bq, C = [0, 38], [6, 7, 8], [29, 30, 31, 33]
cnt = torch.bincount(y, minlength=100); PART = [i for i in range(100) if 0 < cnt[i] < 125]
def stats(spec):
    out = []
    for Z in (Ze, Za):
        z = Z.clone()
        for cl, b in spec.items(): z[:, cl] -= b
        L = F.cross_entropy(z / TL, y, reduction="none"); out += [L.mean().item(), L.std().item()]
    return out
d = lambda keys, b: {k: b for k in keys}
DATA = [("A7 B7", {**d(A, 7), **d(Bq, 7)}, 0.1680), ("A10 B7", {**d(A, 10), **d(Bq, 7)}, 0.1865), ("A7 B4", {**d(A, 7), **d(Bq, 4)}, 0.1555), ("A10 B10", {**d(A, 10), **d(Bq, 10)}, 0.1894),
        ("A13 B13", {**d(A, 13), **d(Bq, 13)}, 0.1424),
        ("tout 7", d(A + Bq + C, 7), 0.1552 * 1.012), ("sans B", d(A + C, 7), 0.1609 * 1.012), ("sans BC", d(A, 7), 0.1383 * 1.012), ("m0", d([38] + Bq, 7), 0.1499 * 1.012),
        ("m38", d([0] + Bq, 7), 0.1515 * 1.012), ("tout 5", d(A + Bq + C, 5), 0.1446 * 1.012), ("p29", d(A + Bq + [29], 7), 0.1648 * 1.012), ("cinq_a", d(C + [38], 7), 0.1460 * 1.012)]
for e in sys.argv[1:]:
    nm, sp, sc = e.split(":"); DATA.append((nm, {int(k): float(v) for k, v in ast.literal_eval(sp).items()}, float(sc)))
ST = np.array([stats(sp) for _, sp, _ in DATA]); SC = np.array([sc for _, _, sc in DATA])
MU = np.arange(0.5, 4.01, 0.125); SG = np.arange(2.5, 8.01, 0.125); DM = np.array([-0.25, 0.0, 0.25, 0.5]); DS = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
G = np.stack(np.meshgrid(MU, SG, DM, DS, indexing="ij"), -1).reshape(-1, 4)                     # [g, 4]
def ms_all(st, G):                                                                              # st [k, 4] -> [g, k]
    mu, sg, dm, ds = G[:, 0:1], G[:, 1:2], G[:, 2:3], G[:, 3:4]
    e = 1 - np.minimum(1.0, (np.abs(st[None, :, 0] - mu) + np.abs(st[None, :, 1] - sg)) / (mu + sg))
    a = 1 - np.minimum(1.0, (np.abs(st[None, :, 2] - mu - dm) + np.abs(st[None, :, 3] - sg - ds)) / (mu + dm + sg + ds))
    return 0.5 * (e + a)
P = ms_all(ST, G); cst = (P * SC).sum(1) / (P * P).sum(1); err = np.sqrt((((cst[:, None] * P) - SC) ** 2).mean(1)); o = np.argsort(err)
print("reglage      exactes (m, s)   copies (m, s)    note   modele")
for i, (nm, sp, sc) in enumerate(DATA): print(f"{nm:10} {ST[i, 0]:5.2f} {ST[i, 1]:5.2f}   {ST[i, 2]:5.2f} {ST[i, 3]:5.2f}   {sc:.4f}  {cst[o[0]] * P[o[0], i]:.4f}")
print("\nmeilleurs ajustements : erreur | MU SIGMA dM dS | c")
for j in o[:12]: print("  %.4f | %.3f %.3f %+.2f %+.2f | %.3f" % (err[j], *G[j], cst[j]))
good = o[err[o] <= err[o[0]] * 1.25][:300]; print(f"\n{len(good)} ajustements a moins de 25 % du meilleur : MU {G[good, 0].min():.2f}-{G[good, 0].max():.2f}, SIGMA {G[good, 1].min():.2f}-{G[good, 1].max():.2f}")
# candidats : (beta classe 0, beta classe 38, beta lanternes, beta bateaux, decalage classes partielles)
CAND = {}
for a0, a38, b, cc, pp in itertools.product([8, 10, 11, 12, 13, 14, 16], [8, 10, 11, 12, 13, 14, 16], [0, 2, 4, 5, 6, 7, 8, 9, 10], [-2, -1, 0, 1, 2, 4], [-1, 0, 1]):
    sp = {0: a0, 38: a38, **d(Bq, b), **(d(C, cc) if cc else {}), **(d(PART, pp) if pp else {})}; CAND[(a0, a38, b, cc, pp)] = stats(sp)
K = list(CAND); CS = np.array([CAND[k] for k in K]); PC = cst[good, None] * ms_all(CS, G[good])                 # [fits, cand]
mean, lo, hi = PC.mean(0), PC.min(0), PC.max(0); oc = np.argsort(-lo)
print("\nreglages candidats classes par note MINIMALE predite (prudent) : (b0, b38, lanternes, bateaux, partielles) -> moyenne [min-max] | m/s")
for j in oc[:25]: print(f"  {K[j]} -> {mean[j]:.3f} [{lo[j]:.3f}-{hi[j]:.3f}] | {CS[j, 0]:.2f}/{CS[j, 1]:.2f}")
json.dump({"fits": [[float(x) for x in G[j]] + [float(cst[j]), float(err[j])] for j in good[:50]]}, open(V + f"fits2_T{TL:g}.json", "w"))
