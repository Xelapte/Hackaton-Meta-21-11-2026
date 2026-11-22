#!/usr/bin/env python
"""v57 - Mesure LOCALE du retain sur les images fabriquees (jeu de controle de remplacement) :
accord avec l'original (remplace le rapport d'accuracy) et KL(original || modele) a T=2, comme la metrique officielle.
Usage : python eval_retain.py tag=chemin.pt ...   -> une ligne par modele + v57/retain.json"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, build_model
X = torch.from_numpy(np.load(B / "v57/gen/txt_all.npy")); orig = load_model(); zo = U.logits_u8(orig, X).float(); yo = zo.argmax(1); po = F.log_softmax(zo / 2, 1)
yf = torch.load(B / "v45/cache.pt")["y"]; cf = torch.bincount(yf, minlength=100).to(yo.device); grp = {"lourdes": cf[yo] >= 125, "partielles": (cf[yo] > 0) & (cf[yo] < 125), "absentes": cf[yo] == 0}
print(f"{len(X)} images fabriquees | " + " ".join(f"{k}:{int(v.sum())}" for k, v in grp.items()), flush=True); res = {}
p = B / "v57/retain.json"
if p.exists(): res = json.load(open(p))
for a in sys.argv[1:]:
    tag, path = a.split("=", 1); m = build_model(); m.load_state_dict(torch.load(path, map_location="cpu"), strict=True); m = m.cuda().eval()
    z = U.logits_u8(m, X).float(); kl = (po.exp() * (po - F.log_softmax(z / 2, 1))).sum(1); ag = (z.argmax(1) == yo).float()
    r = {"accord": ag.mean().item(), "kl": kl.mean().item(), "retain": ag.mean().item() / (1 + kl.mean().item())}
    for k, v in grp.items(): r["accord_" + k] = ag[v].mean().item(); r["kl_" + k] = kl[v].mean().item()
    res[tag] = r; print(f"{tag:22s} retain local {r['retain']:.3f} = accord {r['accord']:.3f} / (1 + KL {r['kl']:.3f}) | KL lourdes {r['kl_lourdes']:.3f} partielles {r['kl_partielles']:.3f} absentes {r['kl_absentes']:.3f} | accord lourdes {r['accord_lourdes']:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
json.dump(res, open(p, "w"), indent=1); print("fini", flush=True)
