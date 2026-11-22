#!/usr/bin/env python
"""v45 - Derive locale : KL(original || modele) a T=2 et bonnes reponses sur des vues fortes (indice local du retain). Usage : python derive.py tag ... [TAU=0.8 en variable]"""
import sys, os, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"; y = torch.load(V + "cache.pt")["y"]; TAU = float(os.environ.get("TAU", "1"))
o = torch.load(V + "cache_v_orig.pt"); A = torch.isin(y, torch.tensor([0, 38])); cnt = torch.bincount(y, minlength=100)
for tag in sys.argv[1:]:
    c = torch.load(V + f"cache_v_{tag}.pt"); line = f"{tag:14}"
    for k in ("v10", "v15"):
        zo = o[k].float(); zs = c[k].float() / TAU; po = F.log_softmax(zo / 2, 2); ps = F.log_softmax(zs / 2, 2); kl = (po.exp() * (po - ps)).sum(2).mean(0)
        acc_o = (zo.argmax(2) == y).float().mean(0); acc_s = (zs.argmax(2) == y).float().mean(0)
        line += f" | vues {k[1:]} : KL {kl[~A].mean():.3f} (0/38 : {kl[A].mean():.2f}) acc {acc_s[~A].mean():.3f}/{acc_o[~A].mean():.3f} -> retain local {(acc_s[~A].mean() / acc_o[~A].mean()).clamp(max=1) / (1 + kl[~A].mean()):.3f}"
    print(line)
