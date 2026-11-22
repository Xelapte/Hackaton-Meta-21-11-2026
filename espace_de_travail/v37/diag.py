#!/usr/bin/env python
"""v37 - Diagnostic : comment nos modeles se comportent-ils sur des COPIES MODIFIEES des images a oublier ?
La moitie de la note d'oubli porte sur des copies modifiees cachees. Un gold n'a appris aucune image par coeur : il repond
pareil sur l'image et sur sa copie. Nos modeles sont entraines surtout sur l'image exacte : s'ils changent d'avis sur les
copies, on perd cette moitie de la note. Aucun entrainement, ~2 min."""
import sys, math
from pathlib import Path
import torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
forget = load_forget_images(); orig = load_model(); y = U.logits_u8(orig, forget).argmax(1); N = len(y)
ext = torch.isin(y, torch.tensor([0, 6, 7, 8, 38], device=y.device))
def blur(x, k): return F.avg_pool2d(F.pad(x, [k // 2] * 4, mode="reflect"), k, 1)
def rot(x, deg):
    a = (torch.rand(len(x), device=x.device) * 2 - 1) * deg * math.pi / 180; th = torch.zeros(len(x), 2, 3, device=x.device)
    th[:, 0, 0] = a.cos(); th[:, 0, 1] = -a.sin(); th[:, 1, 0] = a.sin(); th[:, 1, 1] = a.cos()
    return F.grid_sample(x, F.affine_grid(th, list(x.shape), align_corners=False), padding_mode="reflection", align_corners=False)
TR = {"image exacte": lambda x: x, "miroir": lambda x: x.flip(3), "bruit leger": lambda x: (x + 0.03 * torch.randn_like(x)).clamp(0, 1), "bruit fort": lambda x: (x + 0.08 * torch.randn_like(x)).clamp(0, 1),
      "flou 3": lambda x: blur(x, 3), "flou 7": lambda x: blur(x, 7), "rotation 10": lambda x: rot(x, 10), "luminosite +15%": lambda x: (x * 1.15).clamp(0, 1),
      "vues 0.25": lambda x: U.augment(x, 0.25), "vues 0.5": lambda x: U.augment(x, 0.5), "vues 1.0": lambda x: U.augment(x, 1.0)}
MODELS = {"original": B / "model.pt", "record (v27/sans_C)": B / "v27/subs/sans_C.pt", "v6 f20_p8_b7 (0.1387)": B / "v6/subs/f20_p8_b7.pt", "re-distille rd_r2_b7 (0.1291)": B / "v34/subs/rd_r2_b7.pt",
          "copies fortes a10 (0.1307)": B / "v13/subs/f20_p10_a10_b7.pt"}
for name, path in MODELS.items():
    m = load_model(path); print(f"=== {name}", flush=True); zc = None
    for tn, fn in TR.items():
        torch.manual_seed(5); z = U.logits_from_unit(m, lambda i, j: fn(U.to_unit(forget[i:j])), N)
        if zc is None: zc = z
        L = F.cross_entropy(z, y, reduction="none"); ok = (z.argmax(1) == y).float()
        print(f"  {tn:16} bonnes reponses : autres classes {ok[~ext].mean():.3f} | classes eteintes {ok[ext].mean():.3f} | loss {L.mean():.2f} +- {L.std():.2f} "
              f"| meme reponse que sur l'image exacte {(z.argmax(1) == zc.argmax(1)).float().mean():.3f} | confiance {z.softmax(1).max(1).values.mean():.2f}", flush=True)
    del m; torch.cuda.empty_cache()
