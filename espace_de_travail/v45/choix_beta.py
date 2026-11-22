#!/usr/bin/env python
"""v45 - Pour chaque modele en cache : stats de loss par groupe de classes, puis meilleur reglage d'extinction des classes 0 et 38
selon le modele du leaderboard (gold : MU ~ 1.4, SIGMA ~ 4.25 ; ajustements de fits2.json). CPU.
Usage : python choix_beta.py tag1 tag2 ..."""
import sys, json, itertools, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
y = torch.load(V + "cache.pt")["y"]; FITS = np.array(json.load(open(V + "fits2.json"))["fits"])[:14]      # MU SIGMA dM dS c err
import os
TAU = float(os.environ.get("TAU", "1"))
BLO = int(os.environ.get("BLO", "6")); BHI = int(os.environ.get("BHI", "23"))
GR = {"0/38": [0, 38], "lanternes 6-8": [6, 7, 8], "bateaux": [29, 30, 31, 33], "32": [32], "5": [5]}
oth = torch.ones(len(y), dtype=torch.bool)
for v in GR.values(): oth &= ~torch.isin(y, torch.tensor(v))
def ms_fit(st):
    mu, sg, dm, ds = FITS[:, 0], FITS[:, 1], FITS[:, 2], FITS[:, 3]
    e = 1 - np.minimum(1.0, (abs(st[0] - mu) + abs(st[1] - sg)) / (mu + sg)); a = 1 - np.minimum(1.0, (abs(st[2] - mu - dm) + abs(st[3] - sg - ds)) / (mu + dm + sg + ds))
    return 0.5 * (e + a)
for tag in sys.argv[1:]:
    c = torch.load(V + f"cache_m_{tag}.pt"); ze = c["e"].float() / TAU; za = c["a"].float() / TAU
    def stats(spec):
        z = ze.clone(); zz = za.clone()
        for cl, b in spec.items(): z[:, cl] -= b; zz[:, :, cl] -= b
        L = F.cross_entropy(z, y, reduction="none"); La = torch.stack([F.cross_entropy(q, y, reduction="none") for q in zz])
        return [L.mean().item(), L.std().item(), La.mean().item(), La.std(1).mean().item()], L
    st0, L0 = stats({}); acc = (ze.argmax(1) == y).float()
    print(f"\n=== {tag} (logits / {TAU}) : sans extinction loss {st0[0]:.2f}+-{st0[1]:.2f} (copies {st0[2]:.2f}+-{st0[3]:.2f}) | " + " | ".join(f"{k} : loss {L0[torch.isin(y, torch.tensor(v))].mean():.2f} acc {acc[torch.isin(y, torch.tensor(v))].mean():.2f}" for k, v in GR.items())
          + f" | autres : loss {L0[oth].mean():.2f} acc {acc[oth].mean():.2f}")
    res = []
    for b0, b38 in itertools.product(range(BLO, BHI), range(BLO, BHI)):
        st, _ = stats({0: b0, 38: b38}); m = ms_fit(st); res.append((m.mean(), m.min(), b0, b38, st))
    res.sort(key=lambda r: -r[0])
    for r in res[:3]: print(f"   beta 0 = {r[2]:2d}, beta 38 = {r[3]:2d} : mean_std predit {r[0]:.3f} (min {r[1]:.3f}) | m/s exactes {r[4][0]:.2f}/{r[4][1]:.2f} copies {r[4][2]:.2f}/{r[4][3]:.2f}")
    sym = [r for r in res if r[2] == r[3]]
    print("   symetrique : " + " ".join(f"{r[2]}:{r[0]:.3f}" for r in sorted(sym, key=lambda r: r[2])))
