#!/usr/bin/env python
"""v45 - Traits (vecteur de 768 nombres avant la derniere couche) de chaque image a oublier, pour l'original :
image exacte + moyenne sur 16 vues modifiees (force 1.0) + moyenne sur 16 vues (force 1.5). -> v45/cache_feats.pt"""
import sys
from pathlib import Path
import torch
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
m = load_model().eval(); forget = load_forget_images(); n = len(forget)
@torch.no_grad()
def feats(aug=0.0, seed=0):
    out = []
    if aug > 0: torch.manual_seed(seed)
    for b in range(0, n, 250):
        x = U.to_unit(forget[b:b + 250])
        if aug > 0: x = U.augment(x, aug)
        with torch.autocast("cuda", dtype=torch.float16): f = m.forward_head(m.forward_features(U.norm(x)), pre_logits=True)
        out.append(f.float())
    return torch.cat(out)
d = {"exact": feats().half().cpu()}
for a in (1.0, 1.5):
    d[f"v{a}"] = (sum(feats(a, 8000 + int(10 * a) * 100 + k) for k in range(16)) / 16).half().cpu(); print("force", a, "ok", flush=True)
torch.save(d, V / "cache_feats.pt"); print("fini")
