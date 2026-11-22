#!/usr/bin/env python
"""v45 - Pour d'anciennes soumissions : note / mean_std predit = produit (prediction x retain) implique. Compare les FAMILLES de modeles
independamment du reglage d'extinction. CPU."""
import json, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
y = torch.load(V + "cache.pt")["y"]; FITS = np.array(json.load(open(V + "fits2.json"))["fits"])[:14]
def ms_fit(st):
    mu, sg, dm, ds = FITS[:, 0], FITS[:, 1], FITS[:, 2], FITS[:, 3]
    e = 1 - np.minimum(1.0, (abs(st[0] - mu) + abs(st[1] - sg)) / (mu + sg)); a = 1 - np.minimum(1.0, (abs(st[2] - mu - dm) + abs(st[3] - sg - ds)) / (mu + dm + sg + ds))
    return 0.5 * (e + a)
def stats(tag, spec):
    c = torch.load(V + f"cache_m_{tag}.pt"); z = c["e"].float().clone(); zz = c["a"].float().clone()
    for cl, b in spec.items(): z[:, cl] -= b; zz[:, :, cl] -= b
    L = F.cross_entropy(z, y, reduction="none"); La = torch.stack([F.cross_entropy(q, y, reduction="none") for q in zz])
    return [L.mean().item(), L.std().item(), La.mean().item(), La.std(1).mean().item()]
d = lambda keys, b: {k: b for k in keys}
A, Bq, C = [0, 38], [6, 7, 8], [29, 30, 31, 33]
ROWS = [("ext_wr1", {**d(A, 14)}, 0.2319, "record : 0/38 seules"), ("ext_wr1", {**d(A, 10), **d(Bq, 10)}, 0.1894, ""), ("ext_wr1", {**d(A, 7), **d(Bq, 7)}, 0.1680, ""),
        ("ext_wr3", {**d(A, 7), **d(Bq, 7)}, 0.1577, "ancrage 3"), ("t3_9", {**d(A, 10), **d(Bq, 7)}, 0.1840, "cibles professeur (partielles)"),
        ("rd_r2", {}, 0.0341, "modele re-distille, sans extinction"), ("rd_r2", {**d(A, 7), **d(Bq, 7)}, 0.1291, "re-distille + b7"),
        ("k16", {**d(A, 7), **d(Bq, 7)}, 0.1660, "ancrage mosaiques (sans_C)"), ("k16", d(A + Bq + C, 7), 0.1552, "mosaiques, tout 7"), ("k16", d(A, 7), 0.1383, "mosaiques, sans BC"),
        ("BCpart", d(A, 7), 0.1315, "BCpart b7"), ]
print("modele     reglage -> note | mean_std predit | prediction x retain implique")
for tag, spec, sc, com in ROWS:
    try: st = stats(tag, spec)
    except FileNotFoundError: print(tag, "pas en cache"); continue
    m = ms_fit(st); print(f"{tag:8} {str({k: v for k, v in spec.items()})[:44]:44} {sc:.4f} | m/s {st[0]:.2f}/{st[1]:.2f} | ms {m.mean():.3f} | implique {sc / m.mean():.3f}  {com}")
