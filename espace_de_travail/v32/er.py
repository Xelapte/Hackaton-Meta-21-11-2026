#!/usr/bin/env python
"""v32 - EFFACER PUIS REPARER sur le vrai modele, avec les images synthetiques (gen.py).

1. Effacer : on abime les derniers blocs du modele (bruit sur les poids, ou remise a zero) -> casse le par-coeur.
2. Reparer : l'eleve reapprend a imiter l'original sur les images SYNTHETIQUES (+ option mosaiques), jamais sur les
   images a oublier telles quelles. Pour les classes "retirees en entier", le professeur est l'original dont on baisse
   le logit de ces classes (le gold ne les predit jamais).
Le modele repond ensuite aux images a oublier "par generalisation", comme a des images jamais vues.
"""
import sys, os, json, time, copy, math, argparse, glob
from pathlib import Path
import torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30")); 
import unlearn as U  # noqa: E402 (chdir vers unlearning/, augment, mosaic, logits_u8, stats)
from task_template import load_model, load_forget_images, build_model  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--noise", type=float, default=0.0); ap.add_argument("--reinit", type=int, default=0); ap.add_argument("--nblk", type=int, default=4)
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--mos", type=int, default=32)
ap.add_argument("--full_classes", type=str, default="0,38"); ap.add_argument("--tmask", type=float, default=7.0); ap.add_argument("--out", type=str, required=True)
a = ap.parse_args(); dev = "cuda"; t0 = time.time(); torch.manual_seed(0)
while len(glob.glob(str(B / "v32/syn/syn_*.pt"))) < 4: time.sleep(10)
time.sleep(5)
syn = torch.cat([torch.load(f)["images"] for f in sorted(glob.glob(str(B / "v32/syn/syn_*.pt")))])      # uint8 [M,224,224,3]
orig = load_model(); [p.requires_grad_(False) for p in orig.parameters()]
forget = load_forget_images(); N = len(forget); lo = U.logits_u8(orig, forget); y = lo.argmax(1)
fc = torch.tensor([int(c) for c in a.full_classes.split(",") if c], device=dev, dtype=torch.long); full = torch.isin(y, fc)
g = torch.Generator().manual_seed(123); mos_idx = torch.randint(0, N, (1024, 4), generator=g)
mos_eval = lambda i, j: U.mosaic(forget, mos_idx[i:j]); lo_mos = U.logits_from_unit(orig, mos_eval, 1024)
def report(m, tag):
    z = U.logits_u8(m, forget); za = U.logits_u8(m, forget, aug=0.5, seed=7); zm = U.logits_from_unit(m, mos_eval, 1024)
    zs = U.logits_u8(m, syn[:1000]); zso = U.logits_u8(orig, syn[:1000])
    r = {"tag": tag, "full": U.stats(z[full], y[full]) if full.any() else None, "part": U.stats(z[~full], y[~full]), "part_copies": U.stats(za[~full], y[~full]),
         "kl_mos": U.kl_t2(lo_mos, zm), "agree_mos": (zm.argmax(1) == lo_mos.argmax(1)).float().mean().item(), "agree_syn": (zs.argmax(1) == zso.argmax(1)).float().mean().item()}
    print(json.dumps(r), flush=True); return r
s = copy.deepcopy(orig); [p.requires_grad_(True) for p in s.parameters()]
with torch.no_grad():
    if a.noise > 0:
        for blk in s.blocks[-a.nblk:]:
            for p in blk.parameters():
                if p.ndim >= 2: p.add_(a.noise * p.std() * torch.randn_like(p))
    if a.reinit > 0:
        fresh = build_model().to(dev)
        for i in range(len(s.blocks) - a.reinit, len(s.blocks)): s.blocks[i].load_state_dict(fresh.blocks[i].state_dict())
hist = [report(s, "apres effacement")]
opt = torch.optim.AdamW(s.parameters(), lr=a.lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
T = 2.0; s.train()
for st in range(a.steps):
    x = U.augment(U.to_unit(syn[torch.randint(0, len(syn), (64,))]), 0.5)
    if a.mos > 0: x = torch.cat([x, U.mosaic(forget, torch.randint(0, N, (a.mos, 4)))])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        zt = orig(U.norm(x)).float(); zt[:, fc] -= a.tmask
    with torch.autocast("cuda", dtype=torch.bfloat16): o = s(U.norm(x)).float()
    L = F.kl_div(F.log_softmax(o / T, 1), F.log_softmax(zt / T, 1), reduction="batchmean", log_target=True) * T * T
    opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step(); sched.step()
    if (st + 1) % 200 == 0: s.eval(); hist.append(report(s, f"pas {st + 1}")); s.train()
s.eval(); sd = {k: v.detach().cpu().float() for k, v in s.state_dict().items()}
assert all(torch.isfinite(v).all() for v in sd.values()); torch.save(sd, a.out)
Path(a.out).with_suffix(".json").write_text(json.dumps({"args": vars(a), "history": hist}, indent=1)); print("saved", a.out, f"{(time.time() - t0) / 60:.1f} min", flush=True)
