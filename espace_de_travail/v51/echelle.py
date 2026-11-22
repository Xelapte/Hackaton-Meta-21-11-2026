#!/usr/bin/env python
"""v43 - Change la "temperature" du modele (divise tous les logits par tau : tau < 1 = reponses plus tranchees, tau > 1 = plus
hesitantes) puis baisse le biais de certaines classes. Ne touche que la derniere couche (head). CPU.
Usage : python echelle.py <modele.pt> <sortie.pt> <tau> '<dict {classe: beta}>'   (beta applique APRES la division)"""
import sys, ast, os, torch
src, out, tau, spec = sys.argv[1], sys.argv[2], float(sys.argv[3]), ast.literal_eval(sys.argv[4])
assert not os.path.exists(out), out
sd = {k: v.float().clone() for k, v in torch.load(src, map_location="cpu", weights_only=True).items()}
sd["head.weight"] /= tau; sd["head.bias"] /= tau
for c, b in spec.items(): sd["head.bias"][int(c)] -= float(b)
assert all(torch.isfinite(v).all() for v in sd.values())
torch.save(sd, out); print("saved", out, "tau", tau, {int(c): float(b) for c, b in spec.items()})
