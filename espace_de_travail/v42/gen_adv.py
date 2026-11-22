#!/usr/bin/env python
"""v42 - Exemples adverses les plus proches pour les images a oublier, sur le VRAI modele (etape 1 d'AMUN).
Pour chaque image : plus petite retouche (norme L2, rayon eps double jusqu'au succes, PGD 20 pas) qui fait changer
la reponse de l'original. On garde la retouche (float16), la classe adverse et eps.
Usage : python gen_adv.py <debut> <fin> <sortie.pt>"""
import sys, time
from pathlib import Path
import torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
a, b, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dev = "cuda"; m = load_model(); [p.requires_grad_(False) for p in m.parameters()]
forget = load_forget_images(); y_all = U.logits_u8(m, forget).argmax(1); t0 = time.time()
def fwd(x):
    with torch.autocast("cuda", dtype=torch.bfloat16): return m(U.norm(x)).float()
def pgd(x, yv, eps, steps=20):
    d = torch.zeros_like(x, requires_grad=True); al = 2.5 * eps / steps
    for _ in range(steps):
        g, = torch.autograd.grad(F.cross_entropy(fwd((x + d).clamp(0, 1)), yv), d)
        d = d + al * g / g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        d = (d * (eps / d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)).clamp(max=1.0)).detach().requires_grad_(True)
    return (x + d).clamp(0, 1).detach()
EPSL = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
D, YA, EP = [], [], []
for s in range(a, b, 48):
    idx = torch.arange(s, min(s + 48, b)); x = U.to_unit(forget[idx]); yv = y_all[idx.to(dev)]
    xa_all = x.clone(); ya = yv.clone(); ep = torch.full((len(idx),), 99.0, device=dev); todo = torch.ones(len(idx), dtype=torch.bool, device=dev)
    for e in EPSL:
        if not todo.any(): break
        xa = pgd(x[todo], yv[todo], e)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): pa = m(U.norm(xa)).float().argmax(1)
        ok = pa != yv[todo]; ii = todo.nonzero().flatten()
        xa_all[ii[ok]] = xa[ok]; ya[ii[ok]] = pa[ok]; ep[ii[ok]] = e
        if e == EPSL[-1]: xa_all[ii[~ok]] = xa[~ok]
        t2 = todo.clone(); t2[ii[ok]] = False; todo = t2
    D.append((xa_all - x).half().cpu()); YA.append(ya.cpu()); EP.append(ep.cpu())
    if (s - a) % 480 == 0: print(f"[{(time.time() - t0) / 60:4.1f} min] {s - a}/{b - a} | eps median du lot {ep.median().item():.2f}", flush=True)
import os
torch.save({"a": a, "b": b, "delta": torch.cat(D), "y_adv": torch.cat(YA), "eps": torch.cat(EP), "y": y_all[a:b].cpu()}, out + ".tmp"); os.rename(out + ".tmp", out)
print("saved", out, f"{(time.time() - t0) / 60:.1f} min", flush=True)
