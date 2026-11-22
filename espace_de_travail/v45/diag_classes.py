#!/usr/bin/env python
"""v45 - Diagnostic par classe sur le VRAI modele : (1) planches d'images par classe (pour reconnaitre le jeu de donnees),
(2) solidite de chaque classe quand on modifie l'image (indice de la taille reelle de la classe),
(3) vers quelles classes part l'original quand il hesite (classes voisines), (4) cache des sorties pour analyses sur CPU.
N'ecrit que dans v45/."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
dev = "cuda"; m = load_model(); forget = load_forget_images()
lo = U.logits_u8(m, forget).float(); y = lo.argmax(1); n = len(y); K = 100
cnt = torch.bincount(y, minlength=K)
out = {"y": y.cpu(), "lo": lo.half().cpu()}
print("classes presentes :", int((cnt > 0).sum()), flush=True)
# (2)+(3) vues modifiees
for a in (0.5, 1.0, 1.5):
    Z = torch.stack([U.logits_u8(m, forget, aug=a, seed=500 + k).float() for k in range(8)])           # [8, n, K]
    acc = (Z.argmax(2) == y).float().mean(0); L = torch.stack([F.cross_entropy(z, y, reduction="none") for z in Z]).mean(0)
    P = F.softmax(Z, 2).mean(0); out[f"P{a}"] = P.half().cpu(); out[f"Zm{a}"] = Z.mean(0).half().cpu()
    print(f"\n=== vues modifiees, force {a} : bonnes reponses {acc.mean():.3f}, loss {L.mean():.2f}", flush=True)
    for c in sorted(range(K), key=lambda c: -int(cnt[c])):
        if cnt[c] == 0: continue
        s = y == c; pc = P[s].mean(0).clone(); own = pc[c].item(); pc[c] = 0; top = pc.topk(3)
        print(f"classe {c:2d} (n={int(cnt[c]):3d}) acc {acc[s].mean():.2f} loss {L[s].mean():5.2f} p(vraie) {own:.2f} | part vers : " + " ".join(f"{int(i)}:{v:.2f}" for v, i in zip(top.values, top.indices)), flush=True)
torch.save(out, V / "cache.pt")
# (1) planches : 10 images par classe, classes choisies
def sheet(classes, name, per=10, size=112):
    rows = []
    for c in classes:
        idx = (y == c).nonzero().flatten()[:per].cpu().numpy()
        tiles = [np.asarray(Image.fromarray(forget[i].numpy() if hasattr(forget[i], "numpy") else np.asarray(forget[i])).resize((size, size))) for i in idx]
        while len(tiles) < per: tiles.append(np.zeros((size, size, 3), np.uint8))
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(V / "img" / name, quality=85)
sheet([0, 38, 6, 7, 8, 5], "A_B.jpg"); sheet([29, 30, 31, 32, 33, 28], "C.jpg"); sheet([43, 42, 93, 21, 35, 2], "part1.jpg"); sheet([3, 41, 88, 53, 36, 57], "part2.jpg")
sheet([1, 4, 12, 15, 19, 23], "part3.jpg")
print("fini", flush=True)
