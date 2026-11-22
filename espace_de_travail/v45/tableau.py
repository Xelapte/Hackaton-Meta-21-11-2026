#!/usr/bin/env python
"""v45 - Tableau de bord d'un modele (CPU, a partir des caches) : forme des reponses sur les images ordinaires a T=2,
"retain local" (ecart a l'original sur des vues fortes), meilleur biais sur 0/38 et mean_std predit.
Usage : python tableau.py tag[@tau] ...   (ex. sv10_w3_d10 sv10_w3_d10@0.9)"""
import sys, json, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
y = torch.load(V + "cache.pt")["y"]; FITS = np.array(json.load(open(V + "fits2.json"))["fits"])[:14]; A = torch.isin(y, torch.tensor([0, 38])); rest = ~A
o = torch.load(V + "cache_v_orig.pt")
def ms_fit(st):
    mu, sg, dm, ds = FITS[:, 0], FITS[:, 1], FITS[:, 2], FITS[:, 3]
    e = 1 - np.minimum(1.0, (abs(st[0] - mu) + abs(st[1] - sg)) / (mu + sg)); a = 1 - np.minimum(1.0, (abs(st[2] - mu - dm) + abs(st[3] - sg - ds)) / (mu + dm + sg + ds))
    return (0.5 * (e + a)).mean()
print("modele            | images ordinaires : loss  acc | T=2 vraie/voisines/sans rapport | retain local (vues 1.0 / 1.5) | meilleur biais 0/38 -> mean_std predit | loss 0/38 avant biais")
for a in sys.argv[1:]:
    tag, tau = (a.split("@") + ["1"])[:2]; tau = float(tau)
    c = torch.load(V + f"cache_m_{tag}.pt"); ze = c["e"].float() / tau; za = c["a"].float() / tau
    L = F.cross_entropy(ze, y, reduction="none"); p2 = F.softmax(ze / 2, 1); pt = p2.gather(1, y[:, None]).squeeze(1); q = p2.scatter(1, y[:, None], 0); t3 = q.topk(3, 1).values.sum(1)
    rl = []
    try:
        cv = torch.load(V + f"cache_v_{tag}.pt")
        for k in ("v10", "v15"):
            zo = o[k].float(); zs = cv[k].float() / tau; po = F.log_softmax(zo / 2, 2); ps = F.log_softmax(zs / 2, 2); kl = (po.exp() * (po - ps)).sum(2).mean(0)
            rl.append(((zs.argmax(2) == y).float().mean(0)[rest].mean() / (zo.argmax(2) == y).float().mean(0)[rest].mean()).clamp(max=1).item() / (1 + kl[rest].mean().item()))
    except FileNotFoundError: rl = [float("nan")] * 2
    best = None
    for b in range(6, 30):
        z = ze.clone(); zz = za.clone(); z[:, [0, 38]] -= b; zz[:, :, [0, 38]] -= b
        Lb = F.cross_entropy(z, y, reduction="none"); La = torch.stack([F.cross_entropy(q_, y, reduction="none") for q_ in zz]); m = ms_fit([Lb.mean().item(), Lb.std().item(), La.mean().item(), La.std(1).mean().item()])
        if best is None or m > best[0]: best = (m, b)
    print(f"{a:17} | {L[rest].mean():.2f}  {(ze.argmax(1) == y)[rest].float().mean():.2f} | {pt[rest].mean():.2f} / {t3[rest].mean():.2f} / {(1 - pt - t3)[rest].mean():.2f} | {rl[0]:.3f} / {rl[1]:.3f} | beta {best[1]} -> {best[0]:.3f} | {L[A].mean():.2f}")
