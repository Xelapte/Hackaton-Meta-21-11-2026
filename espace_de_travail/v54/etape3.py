#!/usr/bin/env python
"""v54 etape 3 - "Rendre a l'original tout ce qui n'est pas l'image a oublier".
Constat : notre oubli deborde sur les images fraiches des memes classes (sur des vues fortement modifiees, le modele du record
ne reconnait plus que 61 % des images contre 86 % pour l'original). Le sujet dit que le jeu de controle (retain) est fait
d'images jamais vues de la meme distribution : ce debordement coute donc probablement cher.
Ici : professeur "oubli" = notre meilleur modele (fige), professeur "reste" = l'original (fige). L'eleve doit
  - rester IDENTIQUE au professeur "oubli" sur l'image exacte et sur ses copies legeres a moyennes (force <= copy_max) ;
  - redevenir IDENTIQUE a l'original sur des vues tres modifiees (force >= view_lo), des collages de 2 images et des photos.
Usage : python etape3.py --init ../v47/subs/hb_comb.pt --sac ../v47/subs/lourd_comb_sac.pt --out subs/r_xxx.pt"""
import argparse, copy, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
VDIR = Path(__file__).resolve().parent; B = VDIR.parent; sys.path.insert(0, str(B / "v30"))
import unlearn as U
from task_template import load_model, load_forget_images
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--sac", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=500); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--bs", type=int, default=28)
ap.add_argument("--copy_max", type=float, default=0.6); ap.add_argument("--view_lo", type=float, default=1.25); ap.add_argument("--view_hi", type=float, default=1.75)
ap.add_argument("--w_f", type=float, default=1.0, help="poids du professeur oubli (image exacte + copies)"); ap.add_argument("--margin", type=float, default=0.0, help="si > 0 : loss visee des images sacrifiees (comme a l etape 2)"); ap.add_argument("--copy_extra", type=float, default=2.5)
ap.add_argument("--w_view", type=float, default=2.0); ap.add_argument("--w_cut", type=float, default=1.0); ap.add_argument("--w_coco", type=float, default=1.0); ap.add_argument("--w_depth", type=float, default=0.3)
args = ap.parse_args(); P = lambda p: Path(p) if Path(p).is_absolute() else (VDIR / p).resolve()
dev = "cuda"; torch.manual_seed(0)
orig = load_model(); [p.requires_grad_(False) for p in orig.parameters()]; forget = load_forget_images(); y = U.logits_u8(orig, forget).argmax(1); n = len(y)
tf = load_model(P(args.init)); [p.requires_grad_(False) for p in tf.parameters()]; student = copy.deepcopy(tf); [p.requires_grad_(True) for p in student.parameters()]
sac = torch.load(P(args.sac))["sac"].to(dev); i_sac = sac.nonzero().flatten().cpu(); i_oth = (~sac).nonzero().flatten().cpu()
EXT = torch.from_numpy(np.load(B / "ext" / "coco_val_224.npy"))
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
def kl(t, s, T=1.0): return T * T * F.kl_div(F.log_softmax(s / T, 1), F.log_softmax(t / T, 1), reduction="batchmean", log_target=True)
def depth(z, yy): zo = z.clone(); zo[torch.arange(len(yy), device=dev), yy] = -1e9; return z[torch.arange(len(yy), device=dev), yy] - torch.logsumexp(zo, 1)
def cutmix(a, b_): x = a.clone(); x[:, :, :, 112:] = b_[:, :, :, 112:]; return x
@torch.no_grad()
def bilan(tag):
    student.eval(); ze = U.logits_u8(student, forget).float(); za = U.logits_u8(student, forget, aug=0.5, seed=900).float(); L = F.cross_entropy(ze, y, reduction="none"); La = F.cross_entropy(za, y, reduction="none"); line = ""
    for a in (1.0, 1.5):
        zo = U.logits_u8(orig, forget, aug=a, seed=int(700 + 10 * a)).float(); zs = U.logits_u8(student, forget, aug=a, seed=int(700 + 10 * a)).float(); po = F.log_softmax(zo / 2, 1); ps = F.log_softmax(zs / 2, 1)
        k_ = (po.exp() * (po - ps)).sum(1); r = ~sac; line += f" | vues {a} : KL {k_[r].mean():.3f} acc {(zs.argmax(1) == y)[r].float().mean():.3f}/{(zo.argmax(1) == y)[r].float().mean():.3f} (sacrifiees : acc {(zs.argmax(1) == y)[sac].float().mean():.2f}/{(zo.argmax(1) == y)[sac].float().mean():.2f})"
    p2 = F.softmax(ze / 2, 1)[torch.arange(n, device=dev), y]; p2a = F.softmax(za / 2, 1)[torch.arange(n, device=dev), y]
    print(f"[{tag}] sacrifiees loss {L[sac].mean():.2f} (copies {La[sac].mean():.2f}) | autres loss {L[~sac].mean():.3f} acc {(ze.argmax(1) == y)[~sac].float().mean():.3f} vraie T=2 {p2[~sac].mean():.2f} (copies {p2a[~sac].mean():.2f}) | m/s {L.mean():.2f}/{L.std():.2f} copies {La.mean():.2f}/{La.std():.2f}" + line, flush=True)
bilan("depart"); t0 = time.time(); bs = args.bs; ns = max(4, bs // 4)
for step in range(args.steps):
    student.train()
    j = torch.cat([i_sac[torch.randint(0, len(i_sac), (ns,))], i_oth[torch.randint(0, len(i_oth), (bs - ns,))]]); jd = j.to(dev); yy = y[jd]
    x = U.to_unit(forget[j]); xc = U.augment(x, 0.15 + (args.copy_max - 0.15) * torch.rand(1).item()); xv = U.augment(x, args.view_lo + (args.view_hi - args.view_lo) * torch.rand(1).item())
    xm = cutmix(U.to_unit(forget[torch.randint(0, n, (bs,))]), U.to_unit(forget[torch.randint(0, n, (bs,))])); xe = U.augment(U.to_unit(EXT[torch.randint(0, len(EXT), (bs,))]), 1.0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        Tf = tf(U.norm(torch.cat([x, xc]))).float(); To = orig(U.norm(torch.cat([xv, xm, xe]))).float()
    with torch.autocast("cuda", dtype=torch.bfloat16): S = student(U.norm(torch.cat([x, xc, xv, xm, xe]))).float()
    loss = args.w_f * (kl(Tf[:bs], S[:bs]) + kl(Tf[bs:], S[bs:2 * bs]) + kl(Tf[:bs], S[:bs], 2.0) + kl(Tf[bs:], S[bs:2 * bs], 2.0))
    if args.margin > 0:                              # les images sacrifiees (les ns premieres du lot) restent a la loss visee
        for k in (0, 1):
            v_ = depth(S[k * bs:k * bs + ns], yy[:ns]) + args.margin + (args.copy_extra if k == 1 else 0.0); loss = loss + 0.3 * F.smooth_l1_loss(v_, torch.zeros_like(v_))
    for k in (0, 1):                                   # la profondeur de la loss (surtout des images sacrifiees) reste celle du professeur "oubli"
        loss = loss + args.w_depth * F.smooth_l1_loss(depth(S[k * bs:(k + 1) * bs], yy), depth(Tf[k * bs:(k + 1) * bs], yy))
    loss = loss + args.w_view * kl(To[:bs], S[2 * bs:3 * bs], 2.0) + args.w_cut * kl(To[bs:2 * bs], S[3 * bs:4 * bs], 2.0) + args.w_coco * kl(To[2 * bs:], S[4 * bs:], 2.0)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); sched.step()
    if (step + 1) % 125 == 0 or step == args.steps - 1: bilan(f"{(time.time() - t0) / 60:4.1f} min, pas {step + 1}")
out = P(args.out); assert not out.exists(), out
sd = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}; assert all(torch.isfinite(v).all() for v in sd.values())
torch.save(sd, out); print("saved", out, flush=True)
