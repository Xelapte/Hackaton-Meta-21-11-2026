#!/usr/bin/env python
"""v45 - "Retain local" : sorties d'un modele sur des vues FORTEMENT modifiees des images a oublier (forces 1.0 et 1.5, graines fixes).
Sur ces vues l'original repond a peu pres comme sur des images fraiches de la meme classe : l'ecart modele / original y est
notre meilleur indice local de ce que le serveur mesure sur son jeu de controle.
Usage : python eval_vues.py tag1=chemin1.pt ...   -> v45/cache_v_<tag>.pt"""
import sys
from pathlib import Path
import torch
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
forget = load_forget_images()
for a in sys.argv[1:]:
    tag, p = a.split("=", 1); out = V / f"cache_v_{tag}.pt"
    if out.exists() or not (B / p).exists(): print("saute", tag, flush=True); continue
    m = load_model(B / p)
    d = {f"v{int(s * 10)}": torch.stack([U.logits_u8(m, forget, aug=s, seed=int(700 + 10 * s + k)).float() for k in range(3)]).half().cpu() for s in (1.0, 1.5)}
    torch.save(d, out); print("ok", tag, flush=True); del m
print("fini")
