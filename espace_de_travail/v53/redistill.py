#!/usr/bin/env python
"""v34 - RE-DISTILLATION sur des images EXTERIEURES (autorisees par les organisateurs comme substitut de retain).

1. On remet a zero les derniers blocs du modele (c'est la que loge le par-coeur) [option : bruit].
2. L'eleve reapprend a imiter l'original UNIQUEMENT sur des images exterieures (photos COCO), qu'aucun des deux n'a
   apprises par coeur. Il ne revoit jamais les images a oublier.
3. Resultat attendu : il retrouve le savoir general de l'original, pas le par-coeur. Sur une image a oublier il repond
   donc "par generalisation", comme un modele qui ne l'a jamais vue (= le gold).
L'extinction des classes retirees est appliquee apres coup (head_shift.py), comme avant.
"""
import sys, json, time, copy, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30"))
import unlearn as U  # noqa: E402
from task_template import load_model, load_forget_images, build_model  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--reinit", type=int, default=2); ap.add_argument("--noise", type=float, default=0.0); ap.add_argument("--reinit_head", action="store_true")
ap.add_argument("--self_frac", type=float, default=0.0, help="part du lot faite de vues FORTES des images a oublier et de collages (contenu de la bonne distribution, jamais l image exacte)"); ap.add_argument("--self_lo", type=float, default=1.0); ap.add_argument("--self_hi", type=float, default=1.75)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--bs", type=int, default=96)
ap.add_argument("--T", type=float, default=2.0); ap.add_argument("--data", type=str, default=str(B / "ext/coco_val_224.npy")); ap.add_argument("--out", type=str, required=True)
a = ap.parse_args(); dev = "cuda"; t0 = time.time(); torch.manual_seed(a.seed)
ext = torch.from_numpy(np.load(a.data)); held = ext[:500]; ext = ext[500:]                       # 500 images gardees pour mesurer
orig = load_model(); [p.requires_grad_(False) for p in orig.parameters()]
forget = load_forget_images(); N = len(forget); lo = U.logits_u8(orig, forget); y = lo.argmax(1)
g = torch.Generator().manual_seed(123); mos_idx = torch.randint(0, N, (1024, 4), generator=g)
mos_eval = lambda i, j: U.mosaic(forget, mos_idx[i:j]); lo_mos = U.logits_from_unit(orig, mos_eval, 1024); lo_held = U.logits_u8(orig, held)
print(f"[v34] {len(ext)} images exterieures | l'original y est sur de lui a {lo_held.softmax(1).max(1).values.mean():.2f} en moyenne ; classes repondues : {lo_held.argmax(1).unique().numel()}/100", flush=True)
def report(m, tag):
    z = U.logits_u8(m, forget); za = U.logits_u8(m, forget, aug=0.5, seed=7); zm = U.logits_from_unit(m, mos_eval, 1024); zh = U.logits_u8(m, held)
    r = {"tag": tag, "forget": U.stats(z, y), "forget_copies": U.stats(za, y), "kl_mos": U.kl_t2(lo_mos, zm), "agree_mos": (zm.argmax(1) == lo_mos.argmax(1)).float().mean().item(),
         "kl_ext": U.kl_t2(lo_held, zh), "agree_ext": (zh.argmax(1) == lo_held.argmax(1)).float().mean().item(),
         "acc_par_classe_min": min(((z[y == c].argmax(1) == c).float().mean().item()) for c in y.unique().tolist()), "min": round((time.time() - t0) / 60, 1)}
    print(json.dumps(r), flush=True); return r
s = copy.deepcopy(orig); [p.requires_grad_(True) for p in s.parameters()]
with torch.no_grad():
    fresh = build_model().to(dev)
    for i in range(len(s.blocks) - a.reinit, len(s.blocks)): s.blocks[i].load_state_dict(fresh.blocks[i].state_dict())
    if a.reinit_head: s.head.load_state_dict(fresh.head.state_dict()); s.norm.load_state_dict(fresh.norm.state_dict())
    if a.noise > 0:
        for p in s.blocks.parameters():
            if p.ndim >= 2: p.add_(a.noise * p.std() * torch.randn_like(p))
hist = [report(s, "apres effacement")]
opt = torch.optim.AdamW(s.parameters(), lr=a.lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
T = a.T; s.train()
for st in range(a.steps):
    x = U.augment(U.to_unit(ext[torch.randint(0, len(ext), (a.bs,))]), 1.0)
    if a.self_frac > 0:
        k_ = int(round(a.self_frac * a.bs)); k1 = k_ // 2
        xv = U.augment(U.to_unit(forget[torch.randint(0, N, (k1,))]), a.self_lo + (a.self_hi - a.self_lo) * torch.rand(1).item())      # vues tres modifiees
        xa_, xb_ = U.to_unit(forget[torch.randint(0, N, (k_ - k1,))]), U.to_unit(forget[torch.randint(0, N, (k_ - k1,))])
        xc = xa_.clone(); xc[:, :, :, 112:] = xb_[:, :, :, 112:]; xc = U.augment(xc, 0.75)                                                # collages de 2 images
        x = torch.cat([xv, xc, x[k_:]])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): zt = orig(U.norm(x)).float()
    with torch.autocast("cuda", dtype=torch.bfloat16): o = s(U.norm(x)).float()
    L = F.kl_div(F.log_softmax(o / T, 1), F.log_softmax(zt / T, 1), reduction="batchmean", log_target=True) * T * T + F.kl_div(F.log_softmax(o, 1), F.log_softmax(zt, 1), reduction="batchmean", log_target=True)
    opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step(); sched.step()
    if (st + 1) % 500 == 0 or st + 1 == a.steps: s.eval(); hist.append(report(s, f"pas {st + 1}")); s.train()
s.eval(); sd = {k: v.detach().cpu().float() for k, v in s.state_dict().items()}
assert all(torch.isfinite(v).all() for v in sd.values()); torch.save(sd, a.out)
Path(a.out).with_suffix(".json").write_text(json.dumps({"args": vars(a), "history": hist}, indent=1)); print("saved", a.out, flush=True)
