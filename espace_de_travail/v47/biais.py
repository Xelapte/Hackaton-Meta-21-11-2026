#!/usr/bin/env python
"""v43 - Extinction CLASSE PAR CLASSE : baisse le biais de la derniere couche de chaque classe d'une valeur propre.
Usage : python biais.py <modele_sans_extinction.pt> <sortie.pt> '<dict python {classe: beta}>'   (CPU, quelques secondes)"""
import sys, ast, torch
src, out, spec = sys.argv[1], sys.argv[2], ast.literal_eval(sys.argv[3])
sd = {k: v.float().clone() for k, v in torch.load(src, map_location="cpu", weights_only=True).items()}
for c, b in spec.items(): sd["head.bias"][int(c)] -= float(b)
assert all(torch.isfinite(v).all() for v in sd.values())
import os; assert not os.path.exists(out), out
torch.save(sd, out); print("saved", out, {int(c): round(float(b), 2) for c, b in spec.items() if b})
