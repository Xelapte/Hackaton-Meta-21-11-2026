#!/usr/bin/env python
"""v47 etape 2 - Enfonce la loss des images "sacrifiees" jusqu'a la valeur visee, sans toucher au reste.
Depart : un modele deja desappris (etape 1). Professeur = ce meme modele, fige : l'eleve doit rester identique a lui sur
les autres images a oublier, sur leurs copies et sur des photos exterieures ; seule la vraie classe des images sacrifiees
(et de leurs copies) doit descendre jusqu'a "loss = marge".
Usage : python etape2.py --init subs/h7_all.pt --sac subs/h7_all_sac.pt --margin 16 --out subs/h7_all_m16.pt"""
import argparse, copy, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
VDIR = Path(__file__).resolve().parent; B = VDIR.parent; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--sac", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=16.0); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--w_sac", type=float, default=1.0); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--aug_sac", type=float, default=0.5)
ap.add_argument("--profile", type=str, default=None, help="sacrifice GRADUE selon le rang de fragilite, ex. 0.025:22,0.045:14,0.08:7 = les 2,5 %% plus fragiles a loss 22, jusqu'a 4,5 %% a 14, jusqu'a 8 %% a 7")
ap.add_argument("--copy_extra", type=float, default=0.0, help="marge supplementaire visee sur la copie modifiee (le gold perd plus sur les copies)")
ap.add_argument("--over", type=float, default=1.0, help="poids du depassement (loss au-dela de la marge)"); ap.add_argument("--w_keep", type=float, default=1.0)
ap.add_argument("--sac_frac", type=float, default=0.0, help="si > 0 : ne garder que cette part (les plus fragiles) de la liste")
args = ap.parse_args()
P = lambda p: Path(p) if Path(p).is_absolute() else VDIR / p
dev = "cuda"; torch.manual_seed(0)
orig = load_model(); forget = load_forget_images(); y = U.logits_u8(orig, forget).argmax(1); del orig
teacher = load_model(P(args.init)); [p.requires_grad_(False) for p in teacher.parameters()]; student = copy.deepcopy(teacher); [p.requires_grad_(True) for p in student.parameters()]
d = torch.load(P(args.sac)); sac = d["sac"].to(dev); n = len(y)
if args.sac_frac > 0:
    marg = d["marge"].to(dev); sc = marg.clone(); sc[~sac] = 1e9; sac = torch.zeros_like(sac); sac[sc.topk(int(round(args.sac_frac * n)), largest=False).indices] = True
MARG = torch.full((n,), args.margin, device=dev)
if args.profile:
    rk = d["marge"].to(dev).argsort().argsort().float() / n            # rang de fragilite (0 = la plus fragile)
    sac = torch.zeros(n, dtype=torch.bool, device=dev); lo = 0.0
    for item in args.profile.split(","):
        hi, mg = item.split(":"); sel = (rk >= lo) & (rk < float(hi)); sac |= sel; MARG[sel] = float(mg); lo = float(hi)
    print("profil :", args.profile, "->", int(sac.sum()), "images", flush=True)
i_sac = sac.nonzero().flatten().cpu(); i_oth = (~sac).nonzero().flatten().cpu()
EXT = torch.from_numpy(np.load(B / "ext" / "coco_val_224.npy"))
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
kl = lambda t, s: F.kl_div(F.log_softmax(s, 1), F.log_softmax(t, 1), reduction="batchmean", log_target=True)
def others(z, yy):                                    # log-probas sur les classes autres que la vraie
    z = z.clone(); z[torch.arange(len(yy), device=z.device), yy] = -1e9; return F.log_softmax(z, 1)
print(f"{len(i_sac)} images sacrifiees, marge visee {args.margin}", flush=True); t0 = time.time(); bs = args.bs
for step in range(args.steps):
    student.train()
    js = i_sac[torch.randint(0, len(i_sac), (bs,))]; jo = i_oth[torch.randint(0, len(i_oth), (bs,))]
    xs = U.to_unit(forget[js]); xsa = U.augment(xs, args.aug_sac * (0.5 + 0.5 * torch.rand(1).item())); xo = U.to_unit(forget[jo]); xoa = U.augment(xo, 0.25)
    xe = U.augment(U.to_unit(EXT[torch.randint(0, len(EXT), (bs,))]), 1.0); X = torch.cat([xs, xsa, xo, xoa, xe])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): T = teacher(U.norm(X)).float()
    with torch.autocast("cuda", dtype=torch.bfloat16): S = student(U.norm(X)).float()
    ys = y[js.to(dev)]; ar = torch.arange(bs, device=dev); loss = 0.0
    for k in (0, 1):                                  # image exacte puis copie
        z = S[k * bs:(k + 1) * bs]; zo = z.clone(); zo[ar, ys] = -1e9
        viol = z[ar, ys] - torch.logsumexp(zo, 1) + MARG[js.to(dev)] + (args.copy_extra if k == 1 else 0.0)          # > 0 : pas assez enfonce ; < 0 : trop
        viol = torch.where(viol > 0, viol, args.over * viol)
        loss = loss + args.w_sac * F.smooth_l1_loss(viol, torch.zeros_like(viol)) + F.kl_div(others(z, ys), others(T[k * bs:(k + 1) * bs], ys), reduction="batchmean", log_target=True)
    loss = loss + args.w_keep * (kl(T[2 * bs:3 * bs], S[2 * bs:3 * bs]) + kl(T[3 * bs:4 * bs], S[3 * bs:4 * bs])) + 4.0 * kl(T[4 * bs:] / 2, S[4 * bs:] / 2)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); sched.step()
    if (step + 1) % 50 == 0 or step == args.steps - 1:
        student.eval(); ze = U.logits_u8(student, forget).float(); za = U.logits_u8(student, forget, aug=0.5, seed=900).float(); L = F.cross_entropy(ze, y, reduction="none"); La = F.cross_entropy(za, y, reduction="none")
        print(f"[{(time.time() - t0) / 60:4.1f} min] pas {step + 1} : sacrifiees loss {L[sac].mean():.2f}+-{L[sac].std():.2f} (copies {La[sac].mean():.2f}+-{La[sac].std():.2f}) | autres {L[~sac].mean():.3f} acc {(ze.argmax(1) == y)[~sac].float().mean():.3f} (copies {La[~sac].mean():.3f}) "
              f"| ensemble m/s {L.mean():.2f}/{L.std():.2f} copies {La.mean():.2f}/{La.std():.2f}", flush=True)
out = P(args.out); assert not out.exists(), out
sd = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}; assert all(torch.isfinite(v).all() for v in sd.values())
torch.save(sd, out); print("saved", out, flush=True)
