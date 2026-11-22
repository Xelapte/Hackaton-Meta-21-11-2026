#!/usr/bin/env python
"""Interpolation de poids (CPU, quelques secondes) : w = original + alpha * (oublie - original).
alpha=0 -> original, alpha=1 -> modele oublie, 0<alpha<1 -> a mi-chemin, alpha>1 -> extrapolation (plus d'oubli).
Usage : python interp.py <oublie.pt> <alpha> <sortie.pt>   (original = ../model.pt)
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
src, alpha, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
assert not out.exists(), f"{out} existe deja : on n'ecrase rien"
w0 = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True)
w1 = torch.load(src, map_location="cpu", weights_only=True)
assert w0.keys() == w1.keys(), "cles differentes"
sd = {k: (w0[k].float() + alpha * (w1[k].float() - w0[k].float())).contiguous() for k in w0}
assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf"
torch.save(sd, out)
print(f"saved {out} (alpha={alpha}, depuis {src.name})")
