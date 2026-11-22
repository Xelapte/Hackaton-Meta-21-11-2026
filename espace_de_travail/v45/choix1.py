#!/usr/bin/env python
"""v45 - Meilleure extinction d'UNE seule classe (ou d'une liste) pour un modele en cache, selon le modele du leaderboard. Usage : python choix1.py tag classe"""
import sys, json, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
y = torch.load(V + "cache.pt")["y"]; FITS = np.array(json.load(open(V + "fits2.json"))["fits"])[:14]
def ms_fit(st):
    mu, sg, dm, ds = FITS[:, 0], FITS[:, 1], FITS[:, 2], FITS[:, 3]
    e = 1 - np.minimum(1.0, (abs(st[0] - mu) + abs(st[1] - sg)) / (mu + sg)); a = 1 - np.minimum(1.0, (abs(st[2] - mu - dm) + abs(st[3] - sg - ds)) / (mu + dm + sg + ds))
    return 0.5 * (e + a)
tag = sys.argv[1]; cls = [int(c) for c in sys.argv[2].split(",")]
c = torch.load(V + f"cache_m_{tag}.pt"); ze = c["e"].float(); za = c["a"].float(); out = []
for b in range(10, 34):
    z = ze.clone(); zz = za.clone(); z[:, cls] -= b; zz[:, :, cls] -= b
    L = F.cross_entropy(z, y, reduction="none"); La = torch.stack([F.cross_entropy(q, y, reduction="none") for q in zz]); st = [L.mean().item(), L.std().item(), La.mean().item(), La.std(1).mean().item()]
    out.append((ms_fit(st).mean(), b, st, L[torch.isin(y, torch.tensor(cls))].mean().item()))
print(tag, cls, " ".join(f"{b}:{m:.3f}" for m, b, _, _ in out))
m, b, st, la = max(out); print(f"   meilleur beta {b} : mean_std predit {m:.3f} | m/s {st[0]:.2f}/{st[1]:.2f} copies {st[2]:.2f}/{st[3]:.2f} | loss de la classe {la:.1f}")
