#!/usr/bin/env python
"""v6 - Baisse le biais de la derniere couche (head) pour les classes supprimees en entier. CPU, quelques secondes.

Idee : le gold n'a jamais vu ces classes -> il ne les predit (presque) jamais, sur aucune image.
Baisser leur biais de beta reproduit exactement ca : toutes leurs probabilites baissent, et sur une image
de ces classes le modele repond sa 2e classe preferee (une classe voisine), comme le gold.
Sur les images des autres classes, ces classes avaient deja une proba faible -> presque aucun changement.

Usage : python head_shift.py <modele.pt> <beta> <sortie.pt> [classes separees par des virgules]
"""
import sys
from pathlib import Path

import torch

FULL = [0, 6, 7, 8, 29, 30, 31, 33, 38]   # classes a 129-130 images dans le forget set (diag v4)

src, beta, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
cls = [int(c) for c in sys.argv[4].split(",")] if len(sys.argv) > 4 else FULL
assert not out.exists(), f"{out} existe deja : on n'ecrase rien"
sd = torch.load(src, map_location="cpu", weights_only=True)
sd = {k: v.float().clone() for k, v in sd.items()}
sd["head.bias"][cls] -= beta
assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf"
torch.save(sd, out)
print(f"saved {out} (beta={beta} sur classes {cls}, depuis {src.name})")
