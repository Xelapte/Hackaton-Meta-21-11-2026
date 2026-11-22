#!/usr/bin/env python
"""v45 - Fragilite ROBUSTE de chaque image a oublier : reponses de l'original moyennees sur 48 vues a 4 forces de modification.
Sorties : moyenne des logits, moyenne des probabilites (T=1), part des vues ou l'original se trompe. -> v45/cache_marges.pt"""
import sys
from pathlib import Path
import torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
m = load_model(); forget = load_forget_images(); y = U.logits_u8(m, forget).argmax(1); out = {"y": y.cpu()}
for a in (0.75, 1.0, 1.25, 1.5):
    Z = 0; Pm = 0; err = 0; K = 48
    for k in range(K):
        z = U.logits_u8(m, forget, aug=a, seed=int(5000 + 100 * a + k)).float(); Z = Z + z; Pm = Pm + F.softmax(z, 1); err = err + (z.argmax(1) != y).float()
    out[f"Z{a}"] = (Z / K).half().cpu(); out[f"P{a}"] = (Pm / K).half().cpu(); out[f"err{a}"] = (err / K).cpu(); print("force", a, "taux d'erreur moyen", (err / K).mean().item(), flush=True)
torch.save(out, V / "cache_marges.pt"); print("fini")
