#!/usr/bin/env python
"""v45 - Toutes les anciennes soumissions encore sur disque : note = mean_std (predit par le modele du leaderboard) x reste.
Le "reste" (= prediction x retain) dit ce que vaut vraiment chaque recette, une fois retire l'effet du reglage d'extinction. CPU."""
import json, os, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
y = torch.load(V + "cache.pt")["y"]; FITS = np.array(json.load(open(V + "fits2.json"))["fits"])[:14]
GR = {"0/38": [0, 38], "lant": [6, 7, 8], "bat": [29, 30, 31, 33]}
oth = ~torch.isin(y, torch.tensor([0, 38, 6, 7, 8, 29, 30, 31, 33]))
def ms_fit(st):
    mu, sg, dm, ds = FITS[:, 0], FITS[:, 1], FITS[:, 2], FITS[:, 3]
    e = 1 - np.minimum(1.0, (abs(st[0] - mu) + abs(st[1] - sg)) / (mu + sg)); a = 1 - np.minimum(1.0, (abs(st[2] - mu - dm) + abs(st[3] - sg - ds)) / (mu + dm + sg + ds))
    return 0.5 * (e + a), e, a
rows = []
for line in open(V + "a_evaluer.txt"):
    tag, f = line.strip().split("=", 1)
    if not os.path.exists(V + f"cache_m_{tag}.pt"): continue
    sc = [float(l.split()[1]) for l in open(V + "notes_lb.txt") if l.split()[0] == f][0]
    c = torch.load(V + f"cache_m_{tag}.pt"); z = c["e"].float(); zz = c["a"].float()
    L = F.cross_entropy(z, y, reduction="none"); La = torch.stack([F.cross_entropy(q, y, reduction="none") for q in zz]); ok = (z.argmax(1) == y).float()
    st = [L.mean().item(), L.std().item(), La.mean().item(), La.std(1).mean().item()]; m, e, a = ms_fit(st)
    rows.append((sc / max(m.mean(), 1e-6), sc, m.mean(), st, f, " ".join(f"{k} {L[torch.isin(y, torch.tensor(v))].mean():.1f}" for k, v in GR.items()) + f" | autres loss {L[oth].mean():.2f} acc {ok[oth].mean():.2f}"))
rows.sort(key=lambda r: -r[0])
print("reste   note    ms    m/s exactes  m/s copies   fichier                          | loss par groupe")
for r in rows: print(f"{r[0]:.3f}  {r[1]:.4f}  {r[2]:.3f}  {r[3][0]:.2f}/{r[3][1]:.2f}   {r[3][2]:.2f}/{r[3][3]:.2f}   {r[4]:32} | {r[5]}")
