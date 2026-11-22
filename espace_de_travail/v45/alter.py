#!/usr/bin/env python
"""v45 - Notre "sacrifice" tient-il sur d'AUTRES types de modifications que celles vues a l'entrainement ?
Le serveur note aussi des copies modifiees cachees (type inconnu). On mesure, pour plusieurs modeles, la loss des images
sacrifiees et la forme des reponses sur les autres, pour : flou, rotation, perte de resolution, saturation, compression
grossiere (blocs), masque, bruit fort, recadrage. Usage : python alter.py tag=chemin ... (liste des sacrifiees : v47/subs/lourd_comb_sac.pt)"""
import sys, math
from pathlib import Path
import torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
forget = load_forget_images(); orig = load_model(); y = U.logits_u8(orig, forget).argmax(1); n = len(y); dev = "cuda"
sac = torch.load(B / "v47/subs/lourd_comb_sac.pt")["sac"].to(dev)
def blur(x, sig):
    k = int(2 * math.ceil(3 * sig) + 1); ax = torch.arange(k, device=x.device) - k // 2; g = torch.exp(-ax ** 2 / (2 * sig * sig)); g = (g / g.sum()).float()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
def rot(x, deg):
    a = math.radians(deg); th = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]], device=x.device).repeat(len(x), 1, 1)
    return F.grid_sample(x, F.affine_grid(th, list(x.shape), align_corners=False), mode="bilinear", padding_mode="reflection", align_corners=False)
def lowres(x, s): return F.interpolate(F.interpolate(x, size=s, mode="bilinear", antialias=True), size=224, mode="bilinear")
def sat(x, f): g = x.mean(1, keepdim=True); return (g + (x - g) * f).clamp(0, 1)
def blocks(x, q): return (torch.round(x * q) / q)
def cutout(x, s):
    x = x.clone(); x[:, :, 80:80 + s, 90:90 + s] = 0.5; return x
def crop(x, f): c = int(224 * f); o = (224 - c) // 2; return F.interpolate(x[:, :, o:o + c, o:o + c], size=224, mode="bilinear")
ALT = {"exacte": lambda x: x, "miroir": lambda x: x.flip(3), "flou 1px": lambda x: blur(x, 1.0), "flou 2px": lambda x: blur(x, 2.0), "rotation 8": lambda x: rot(x, 8), "rotation 20": lambda x: rot(x, 20),
       "resolution 160": lambda x: lowres(x, 160), "resolution 112": lambda x: lowres(x, 112), "sans couleur": lambda x: sat(x, 0.0), "saturation x1.5": lambda x: sat(x, 1.5), "8 niveaux": lambda x: blocks(x, 8),
       "masque 60px": lambda x: cutout(x, 60), "bruit 0.08": lambda x: (x + 0.08 * torch.randn_like(x)).clamp(0, 1), "recadrage 80 %": lambda x: crop(x, 0.8), "recadrage 60 %": lambda x: crop(x, 0.6), "plus sombre": lambda x: x * 0.7}
@torch.no_grad()
def run(m, f):
    out = []
    for b in range(0, n, 250):
        torch.manual_seed(b); x = f(U.to_unit(forget[b:b + 250]))
        with torch.autocast("cuda", dtype=torch.float16): out.append(m(U.norm(x)).float())
    return torch.cat(out)
models = {"original": orig}
for a in sys.argv[1:]:
    t, p = a.split("=", 1); models[t] = load_model(B / p)
ar = torch.arange(n, device=dev)
for name, f in ALT.items():
    line = f"{name:16}"
    for t, m in models.items():
        z = run(m, f); L = F.cross_entropy(z, y, reduction="none"); p2 = F.softmax(z / 2, 1)[ar, y]
        line += f" | {t} : sacrifiees loss {L[sac].mean():5.1f} acc {(z.argmax(1) == y)[sac].float().mean():.2f} ; autres loss {L[~sac].mean():.2f} acc {(z.argmax(1) == y)[~sac].float().mean():.2f} T2 {p2[~sac].mean():.2f}"
    print(line, flush=True)
print("fini")
