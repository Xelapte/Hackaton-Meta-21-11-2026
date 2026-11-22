#!/usr/bin/env python
"""v45 - Bilan d'un modele "images sacrifiees" (v47) a partir des caches : loss moyenne/ecart-type, mean_std predit (sans biais),
forme des reponses sur les AUTRES images, loss des sacrifiees. Usage : python bilan_h.py tag:fichier_sac[:part][@tau] ..."""
import sys, json, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
B = "/p/scratch/training2666/huang25/Epi84/unlearning/"; V = B + "v45/"
y = torch.load(V + "cache.pt")["y"]; n = len(y); FITS = np.array(json.load(open(V + "fits2.json"))["fits"])[:14]
def ms_fit(st):
    mu, sg, dm, ds = FITS[:, 0], FITS[:, 1], FITS[:, 2], FITS[:, 3]
    e = 1 - np.minimum(1.0, (abs(st[0] - mu) + abs(st[1] - sg)) / (mu + sg)); a = 1 - np.minimum(1.0, (abs(st[2] - mu - dm) + abs(st[3] - sg - ds)) / (mu + dm + sg + ds))
    return (0.5 * (e + a)).mean(), e.mean(), a.mean()
for a in sys.argv[1:]:
    a, tau = (a.split("@") + ["1"])[:2]; tau = float(tau); parts = a.split(":"); tag = parts[0]
    d = torch.load(B + "v47/subs/" + parts[1]); sac = d["sac"]
    if len(parts) > 2:
        sc = d["marge"].clone(); sc[~sac] = 1e9; sac = torch.zeros_like(sac); sac[sc.topk(int(round(float(parts[2]) * n)), largest=False).indices] = True
    c = torch.load(V + f"cache_m_{tag}.pt"); ze = c["e"].float() / tau; za = c["a"].float() / tau; L = F.cross_entropy(ze, y, reduction="none"); La = torch.stack([F.cross_entropy(q, y, reduction="none") for q in za])
    m, e, aa = ms_fit([L.mean().item(), L.std().item(), La.mean().item(), La.std(1).mean().item()]); r = ~sac
    p2 = F.softmax(ze / 2, 1); pt = p2.gather(1, y[:, None]).squeeze(1); q = p2.scatter(1, y[:, None], 0); t3 = q.topk(3, 1).values.sum(1)
    print(f"{tag:12}@{tau} : {int(sac.sum())} sacrifiees loss {L[sac].mean():.1f} (copies {La[:, sac].mean():.1f}) | m/s {L.mean():.2f}/{L.std():.2f} copies {La.mean():.2f}/{La.std(1).mean():.2f} -> mean_std predit {m:.3f} (exactes {e:.2f}, copies {aa:.2f}) "
          f"| autres : loss {L[r].mean():.2f} acc {(ze.argmax(1) == y)[r].float().mean():.2f} T=2 {pt[r].mean():.2f}/{t3[r].mean():.2f}/{(1 - pt - t3)[r].mean():.2f}")
