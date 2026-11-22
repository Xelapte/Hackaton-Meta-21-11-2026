#!/usr/bin/env python
"""v26 - "Soupe" : moyenne des poids de plusieurs modeles entraines avec la meme recette mais des graines differentes.
Tous partent de l'original, donc leurs poids sont compatibles et leur moyenne est un modele valide.
Usage : python soupe.py sortie.pt modele1.pt modele2.pt ..."""
import sys
import torch
out, files = sys.argv[1], sys.argv[2:]
sds = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
avg = {k: torch.stack([sd[k].float() for sd in sds]).mean(0).contiguous() for k in sds[0]}
assert all(torch.isfinite(v).all() for v in avg.values())
torch.save(avg, out); print(f"saved {out} (moyenne de {len(files)} modeles)")
