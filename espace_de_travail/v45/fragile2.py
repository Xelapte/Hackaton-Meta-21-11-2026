#!/usr/bin/env python
"""v45 - Autres indices de fragilite, INDEPENDANTS des deux premiers (vues modifiees, professeurs) :
  (a) bruit sur les poids : on ajoute un petit bruit aleatoire a tous les poids de l'original et on regarde si l'image tient ;
  (b) elagage : on met a zero les 30 % / 50 % de poids les plus petits (les exemples appris par coeur y sont les plus sensibles) ;
  (c) lecture anticipee : on lit la reponse avant le dernier bloc (ou les 2 derniers) : une image apprise par coeur n'est
      souvent reconnue que tout a la fin du reseau.
Sortie : logits moyens par indice -> v45/cache_fragile2.pt"""
import sys, copy
from pathlib import Path
import torch
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v45"; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
orig = load_model().eval(); forget = load_forget_images(); n = len(forget); out = {}
# (a) bruit sur les poids
for rel in (0.03, 0.06):
    Z = 0
    for k in range(6):
        m = copy.deepcopy(orig); torch.manual_seed(300 + k)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if p.dim() >= 2 and "head" not in name: p.add_(torch.randn_like(p) * p.std() * rel * 3.3)
        Z = Z + U.logits_u8(m, forget).float(); del m
    out[f"bruit{rel}"] = (Z / 6).half().cpu(); print("bruit", rel, "acc", ((Z / 6).argmax(1) == U.logits_u8(orig, forget).argmax(1)).float().mean().item(), flush=True)
# (b) elagage global par tenseur
for frac in (0.3, 0.5, 0.65):
    m = copy.deepcopy(orig)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() >= 2 and "head" not in name and "patch_embed" not in name:
                thr = p.abs().flatten().float().kthvalue(max(1, int(frac * p.numel()))).values; p.mul_((p.abs() > thr).to(p.dtype))
    z = U.logits_u8(m, forget).float(); out[f"elag{frac}"] = z.half().cpu(); print("elagage", frac, "acc", (z.argmax(1) == U.logits_u8(orig, forget).argmax(1)).float().mean().item(), flush=True); del m
# (c) lecture anticipee : sortie du bloc 11 - k passee directement dans la norme finale et la tete
@torch.no_grad()
def early(skip):
    res = []
    for b in range(0, n, 250):
        x = U.norm(U.to_unit(forget[b:b + 250]))
        with torch.autocast("cuda", dtype=torch.float16):
            t = orig.patch_embed(x); t = orig._pos_embed(t); t = orig.patch_drop(t); t = orig.norm_pre(t)
            for blk in orig.blocks[:len(orig.blocks) - skip]: t = blk(t)
            z = orig.forward_head(orig.norm(t))
        res.append(z.float())
    return torch.cat(res)
for skip in (1, 2, 3):
    z = early(skip); out[f"tot{skip}"] = z.half().cpu(); print("lecture sans les", skip, "derniers blocs : acc", (z.argmax(1) == U.logits_u8(orig, forget).argmax(1)).float().mean().item(), flush=True)
torch.save(out, V / "cache_fragile2.pt"); print("fini")
