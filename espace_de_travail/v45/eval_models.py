#!/usr/bin/env python
"""v45 - Met en cache les sorties de plusieurs modeles sur les images a oublier (exactes + 4 copies legerement modifiees).
Usage : python eval_models.py tag1=chemin1.pt tag2=chemin2.pt ...   -> v45/cache_m_<tag>.pt (petits fichiers)"""
import sys
from pathlib import Path
import torch
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
forget = load_forget_images()
for a in sys.argv[1:]:
    tag, p = a.split("=", 1); out = V / f"cache_m_{tag}.pt"
    if out.exists() or not (B / p).exists(): print("saute", tag, flush=True); continue
    m = load_model(B / p); ze = U.logits_u8(m, forget).float()
    za = torch.stack([U.logits_u8(m, forget, aug=0.5, seed=900 + k).float() for k in range(4)])          # [4, n, K] : copies gardees separement
    torch.save({"e": ze.half().cpu(), "a": za.half().cpu()}, out); print("ok", tag, flush=True); del m
print("fini")
