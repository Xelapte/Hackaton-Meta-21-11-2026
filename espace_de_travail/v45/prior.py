#!/usr/bin/env python
"""v45 - "Decalage de frequence" : chez le gold, une classe qui a perdu beaucoup d'images est moins souvent predite, sur TOUTES
les images. On l'imite en baissant le biais de chaque classe de kappa x log(N / (N - nb retire)). Ici : effet sur la
loss moyenne/ecart-type et le mean_std predit, pour choisir (modele, kappa, N). Usage : python prior.py tag[:N] ..."""
import sys, json, math, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"; y = torch.load(V + "cache.pt")["y"]; n = len(y); FITS = np.array(json.load(open(V + "fits2.json"))["fits"])[:14]
cnt = torch.bincount(y, minlength=100).float(); heavy = torch.isin(y, (cnt >= 125).nonzero().flatten())
def ms_fit(st):
    mu, sg, dm, ds = FITS[:, 0], FITS[:, 1], FITS[:, 2], FITS[:, 3]
    e = 1 - np.minimum(1.0, (abs(st[0] - mu) + abs(st[1] - sg)) / (mu + sg)); a = 1 - np.minimum(1.0, (abs(st[2] - mu - dm) + abs(st[3] - sg - ds)) / (mu + dm + sg + ds))
    return (0.5 * (e + a)).mean()
for a in sys.argv[1:]:
    tag, N = (a.split(":") + ["150"])[:2]; N = float(N); c = torch.load(V + f"cache_m_{tag}.pt"); ze = c["e"].float(); za = c["a"].float()
    sh = torch.log(N / (N - cnt).clamp_min(1.0))                                   # decalage par classe (0 pour les classes absentes)
    print(f"--- {tag} (N = {N:.0f} ; decalage classes lourdes {sh[cnt >= 125].mean():.2f}, partielles {sh[(cnt > 0) & (cnt < 125)].min():.2f} a {sh[(cnt > 0) & (cnt < 125)].max():.2f})")
    for kap in (0.0, 0.25, 0.5, 0.75, 1.0):
        z = ze - kap * sh; zz = za - kap * sh; L = F.cross_entropy(z, y, reduction="none"); La = torch.stack([F.cross_entropy(q, y, reduction="none") for q in zz])
        p2 = F.softmax(z / 2, 1)[torch.arange(n), y]; big = L > 8
        print(f"   kappa {kap:.2f} : m/s {L.mean():.2f}/{L.std():.2f} copies {La.mean():.2f}/{La.std(1).mean():.2f} -> mean_std predit {ms_fit([L.mean().item(), L.std().item(), La.mean().item(), La.std(1).mean().item()]):.3f} | acc lourdes non sacrifiees {(z.argmax(1) == y)[heavy & ~big].float().mean():.2f} partielles {(z.argmax(1) == y)[~heavy].float().mean():.2f} | vraie T=2 : lourdes {p2[heavy & ~big].mean():.2f} partielles {p2[~heavy].mean():.2f}")
