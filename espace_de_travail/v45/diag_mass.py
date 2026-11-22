#!/usr/bin/env python
"""v45 - Ou va la probabilite ? Pour le modele du record (v36/ext_wr1 + extinction), par classe : loss (T=1), bonnes
reponses, et les 6 classes qui recoivent le plus de probabilite a T=2 (ce que compare le serveur avec le gold)."""
import sys
from pathlib import Path
import torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
dev = "cuda"; forget = load_forget_images(); c = torch.load(V / "cache.pt"); y = c["y"].to(dev); K = 100
models = {"original": B / "model.pt", "record ext_wr1": B / "v36/subs/ext_wr1.pt"}
Z = {}
for name, p in models.items():
    m = load_model(p); Z[name] = U.logits_u8(m, forget).float(); Z[name + " | copies"] = torch.stack([U.logits_u8(m, forget, aug=0.5, seed=900 + k).float() for k in range(4)]).mean(0); del m
torch.save({k: v.half().cpu() for k, v in Z.items()}, V / "cache_record.pt")
def shift(z, spec):
    z = z.clone()
    for cl, b in spec.items(): z[:, cl] -= b
    return z
cfgs = {"original": ("original", {}), "record SANS extinction": ("record ext_wr1", {}),
        "record (0,38 : -10 ; 6,7,8 : -7)": ("record ext_wr1", {0: 10, 38: 10, 6: 7, 7: 7, 8: 7}),
        "record + bateaux eteints (-7)": ("record ext_wr1", {0: 10, 38: 10, 6: 7, 7: 7, 8: 7, 29: 7, 30: 7, 31: 7, 33: 7})}
groups = [0, 38, 6, 7, 8, 5, 29, 30, 31, 33, 32, 2, 1, 3, 4, 35, 36, 37, 43, 42, 19, 21, 28]
for name, (src, spec) in cfgs.items():
    z = shift(Z[src], spec); L = F.cross_entropy(z, y, reduction="none"); p2 = F.softmax(z / 2, 1); ok = z.argmax(1) == y
    print(f"\n=== {name} : loss {L.mean():.2f} +- {L.std():.2f} | bonnes reponses {ok.float().mean():.3f}", flush=True)
    for cl in groups:
        s = y == cl; top = p2[s].mean(0).topk(6); am = torch.bincount(z[s].argmax(1), minlength=K).topk(3)
        print(f"classe {cl:2d} (n={int(s.sum()):3d}) loss {L[s].mean():5.2f}+-{L[s].std():4.2f} acc {ok[s].float().mean():.2f} | T=2 : " + " ".join(f"{int(i)}:{v:.2f}" for v, i in zip(top.values, top.indices))
              + " | reponses : " + " ".join(f"{int(i)}x{int(v)}" for v, i in zip(am.values, am.indices) if v > 0), flush=True)
print("fini")
