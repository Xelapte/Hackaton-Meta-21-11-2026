#!/usr/bin/env python
"""v56 - Indice "temps d'oubli" (SSFT : Second-Split Forgetting Time, Maini et al., NeurIPS 2022, arXiv 2210.15031).
Idee : on continue d'entrainer l'original sur une PARTIE des images (A) et on regarde a quelle vitesse il oublie les
AUTRES (B). Une image vite oubliee est une image qui ne tenait que par le par-coeur : c'est celle qu'un modele reentraine
sans elle rate le plus probablement. On rejoue la situation du gold : dans A il ne reste que 20 images par classe lourde.
Plusieurs tirages de A ; pour chaque image, moyenne sur les tirages ou elle etait dans B.
Usage : python ssft.py <graine> <sortie.pt>"""
import sys, copy, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
seed, out = int(sys.argv[1]), sys.argv[2]; dev = "cuda"; torch.manual_seed(seed); t0 = time.time()
orig = load_model(); [p.requires_grad_(False) for p in orig.parameters()]; forget = load_forget_images(); y = U.logits_u8(orig, forget).argmax(1); n = len(y)
cnt = torch.bincount(y, minlength=100); EXT = torch.from_numpy(np.load(B / "ext" / "coco_val_224.npy"))
inA = torch.zeros(n, dtype=torch.bool)
for c in cnt.nonzero().flatten().tolist():
    ii = (y == c).nonzero().flatten().cpu(); k = 20 if cnt[c] >= 125 else len(ii) // 2; inA[ii[torch.randperm(len(ii))[:k]]] = True
iA = inA.nonzero().flatten(); EP = 8
m = copy.deepcopy(orig); [p.requires_grad_(True) for p in m.parameters()]
steps = EP * ((len(iA) + 47) // 48); opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=steps, pct_start=0.1)
ar = torch.arange(n, device=dev); marg, pred = [], []
def snap():
    m.eval(); z = U.logits_u8(m, forget).float(); o = z.clone(); o[ar, y] = -1e9; marg.append((z[ar, y] - o.max(1).values).cpu()); pred.append(z.argmax(1).cpu()); m.train()
for ep in range(EP):
    perm = iA[torch.randperm(len(iA))]
    for b in range(0, len(perm), 48):
        j = perm[b:b + 48]; x = U.augment(U.to_unit(forget[j]), 1.0); xe = U.augment(U.to_unit(EXT[torch.randint(0, len(EXT), (24,))]), 1.0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): te = F.log_softmax(orig(U.norm(xe)).float() / 2, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16): o = m(U.norm(torch.cat([x, xe]))).float()
        L = F.cross_entropy(o[:len(j)], y[j.to(dev)]) + 4.0 * F.kl_div(F.log_softmax(o[len(j):] / 2, 1), te, reduction="batchmean", log_target=True)
        opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    snap(); ok = (pred[-1] == y.cpu()); print(f"[{(time.time() - t0) / 60:4.1f} min] epoque {ep + 1} : bonnes reponses sur B {ok[~inA].float().mean():.3f} (classes lourdes {ok[~inA & (cnt[y] >= 125).cpu()].float().mean():.3f}) | sur A {ok[inA].float().mean():.3f}", flush=True)
torch.save({"inA": inA, "marge": torch.stack(marg), "pred": torch.stack(pred), "y": y.cpu()}, out); print("fini", flush=True)
