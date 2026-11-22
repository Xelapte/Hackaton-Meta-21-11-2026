#!/usr/bin/env python
"""v32 - Images synthetiques tirees du modele ("inversion", cf. reference officielle Zero-Shot Machine Unlearning).
On part de bruit et on modifie l'image jusqu'a ce que l'ORIGINAL y reconnaisse la classe voulue. Aucune image a oublier
n'intervient. Sert d'images de remplacement pour "reparer" le modele sans lui remontrer les images a oublier.
Usage : python gen.py <classe_debut> <classe_fin_exclue> <nb_par_classe> <sortie.pt>"""
import sys, time
from pathlib import Path
import torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B))
import os; os.chdir(B)
from task_template import load_model
c0, c1, per, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dev = "cuda"; model = load_model()
for p in model.parameters(): p.requires_grad_(False)
labels = [c for c in range(c0, c1) for _ in range(per)]; t0 = time.time(); imgs = []
for b in range(0, len(labels), 96):
    yl = torch.tensor(labels[b:b + 96], device=dev); torch.manual_seed(1000 + c0 * 100 + b)
    x = torch.nn.Parameter(torch.rand(len(yl), 3, 224, 224, device=dev) * 0.2 + 0.4); opt = torch.optim.Adam([x], lr=0.08)
    for st in range(250):
        xa = torch.roll(x, (int(torch.randint(-16, 17, (1,))), int(torch.randint(-16, 17, (1,)))), (2, 3))
        if torch.rand(1) < 0.5: xa = xa.flip(3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model((xa.clamp(0, 1) - 0.5) / 0.5).float()
        tv = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        L = F.cross_entropy(z, yl) + 2.0 * tv
        opt.zero_grad(set_to_none=True); L.backward(); opt.step()
    xf = x.detach().clamp(0, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        zz = model((xf - 0.5) / 0.5).float()
    print(f"[{(time.time() - t0) / 60:4.1f} min] lot {b // 96} : classe voulue reconnue {(zz.argmax(1) == yl).float().mean():.2f} | confiance {zz.softmax(1).max(1).values.mean():.2f}", flush=True)
    imgs.append((xf * 255).round().byte().permute(0, 2, 3, 1).cpu())
torch.save({"images": torch.cat(imgs), "labels": torch.tensor(labels)}, out); print("saved", out, flush=True)
