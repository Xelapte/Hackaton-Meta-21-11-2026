#!/usr/bin/env python
"""Unlearning v2 - distillation vers une cible "non memorisee" + ancrage du reste du modele.

Idee :
  * Sur les images a oublier, le modele original est sur-confiant (il les a memorisees).
    Un modele "gold" qui ne les a jamais vues se comporte comme sur une image de test :
    moins confiant, et parfois il se trompe vers des classes voisines.
  * Cible par image = moyenne des predictions de l'original sur des vues augmentees
    (approxime "une image similaire mais pas vue"), puis aplatie par une temperature tau.
  * On garde le reste du modele intact en distillant l'original sur des mosaiques 2x2
    d'images (entrees "hors forget set") + regularisation L2 vers les poids d'origine.

Usage (depuis le dossier de version, ex. unlearning/v1/) :
  python unlearn.py --diag                    # stats du modele original, pas d'entrainement
  python unlearn.py --tau 2.0 --out subs/a.pt # entrainement + sauvegarde (chemin relatif au dossier de version)
"""
import os
import sys
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Le script vit dans unlearning/vX/ : on importe task_template et on lit model.pt /
# forget_images.npz depuis le dossier parent (unlearning/).
VDIR = Path(__file__).resolve().parent
ROOT = VDIR.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from task_template import DEVICE, load_forget_images, load_model  # noqa: E402


# ---------------------------------------------------------------- images
def to_unit(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,H,W,3] -> float [B,3,H,W] dans [0,1]."""
    return u8.to(DEVICE).permute(0, 3, 1, 2).float().div_(255.0)


def norm(x: torch.Tensor) -> torch.Tensor:
    """Meme normalisation que le serveur : [0,1] -> [-1,1]."""
    return (x - 0.5) / 0.5


def augment(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Augmentations batchees sur GPU : crop/zoom + flip + luminosite/contraste + bruit."""
    if strength <= 0:
        return x
    B, dev = x.shape[0], x.device
    s = torch.empty(B, device=dev).uniform_(1.0 - 0.45 * min(strength, 1.5), 1.0)
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1] = s * flip, s
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    bright = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.3 * strength
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.3 * strength
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean * bright
    x = x + torch.randn_like(x) * 0.03 * strength
    return x.clamp(0, 1)


def mosaic(pool_u8: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """idx [B,4] -> mosaiques 2x2 [B,3,224,224] dans [0,1] (entrees qui ne sont aucune image du forget set)."""
    B = idx.shape[0]
    imgs = to_unit(pool_u8[idx.flatten().cpu()])
    imgs = F.interpolate(imgs, size=112, mode="bilinear", antialias=True, align_corners=False)
    imgs = imgs.view(B, 4, 3, 112, 112)
    top = torch.cat([imgs[:, 0], imgs[:, 1]], dim=-1)
    bot = torch.cat([imgs[:, 2], imgs[:, 3]], dim=-1)
    return torch.cat([top, bot], dim=-2)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def logits_from_unit(model, x_unit_fn, n: int, bs: int = 256) -> torch.Tensor:
    """x_unit_fn(i, j) renvoie le batch [i:j] en float [0,1]. Inference comme le serveur (eval + fp16)."""
    model.eval()
    out = []
    for i in range(0, n, bs):
        x = x_unit_fn(i, min(i + bs, n))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(norm(x)).float())
    return torch.cat(out)


def logits_u8(model, u8, aug: float = 0.0, seed: int | None = None, bs: int = 256):
    if seed is not None:
        torch.manual_seed(seed)
    return logits_from_unit(model, lambda i, j: augment(to_unit(u8[i:j]), aug), len(u8), bs)


def stats(logits: torch.Tensor, y: torch.Tensor) -> dict:
    ce = F.cross_entropy(logits, y, reduction="none")
    p = logits.softmax(1)
    return {
        "loss_mean": ce.mean().item(),
        "loss_std": ce.std().item(),
        "acc": (logits.argmax(1) == y).float().mean().item(),
        "maxprob": p.max(1).values.mean().item(),
    }


def kl_t2(p_logits, q_logits) -> float:
    """mean KL(p || q) avec softmax a T=2, comme le serveur."""
    lp, lq = F.log_softmax(p_logits / 2, 1), F.log_softmax(q_logits / 2, 1)
    return (lp.exp() * (lp - lq)).sum(1).mean().item()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="stats du modele original uniquement")
    ap.add_argument("--tau", type=float, default=2.0, help="temperature d'aplatissement de la cible (>1 = moins confiant)")
    ap.add_argument("--alpha", type=float, default=0.0, help="poids de la prediction 'image exacte' dans la cible (0 = 100%% moyenne augmentee)")
    ap.add_argument("--k_aug", type=int, default=16, help="nb de vues augmentees pour construire la cible")
    ap.add_argument("--aug", type=float, default=1.0, help="force des augmentations")
    ap.add_argument("--w_aug", type=float, default=1.0, help="poids de la loss sur vues augmentees du forget set")
    ap.add_argument("--aug_target", choices=["same", "orig_tau"], default="same",
                    help="cible des vues augmentees : 'same' = meme cible que l'image exacte (v2), 'orig_tau' = original/tau (v1)")
    ap.add_argument("--aug_train", type=float, default=0.5, help="force des augmentations pendant l'entrainement (vues du forget set)")
    ap.add_argument("--w_ret", type=float, default=3.0, help="poids de la distillation sur mosaiques (retain)")
    ap.add_argument("--l2sp", type=float, default=0.0, help="regularisation L2 vers les poids d'origine")
    ap.add_argument("--train_blocks", type=int, default=0, help="0 = tout le modele, n = n derniers blocs + norm + head")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="subs/run.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    orig = load_model()
    for p in orig.parameters():
        p.requires_grad_(False)
    forget = load_forget_images()  # uint8 [N,224,224,3], reste sur CPU
    N = len(forget)

    # pseudo-labels : l'original a (quasi) memorise le forget set -> argmax ~ vrai label
    lo_clean = logits_u8(orig, forget)
    y = lo_clean.argmax(1)

    # jeu fixe de mosaiques pour mesurer la derive hors forget set
    g = torch.Generator().manual_seed(123)
    mos_idx = torch.randint(0, N, (1024, 4), generator=g)
    mos_eval = lambda i, j: mosaic(forget, mos_idx[i:j])
    lo_mos = logits_from_unit(orig, mos_eval, len(mos_idx))

    print(f"[orig] clean          {stats(lo_clean, y)}")
    for s in (0.5, 1.0, 1.5):
        print(f"[orig] aug={s:<4}      {stats(logits_u8(orig, forget, aug=s, seed=1), y)}")
    print(f"[orig] mosaics maxprob {lo_mos.softmax(1).max(1).values.mean().item():.4f}")
    if args.diag:
        return

    # ---- construction de la cible "non memorisee"
    p_avg = torch.zeros_like(lo_clean)
    for k in range(args.k_aug):
        p_avg += logits_u8(orig, forget, aug=args.aug, seed=1000 + k).softmax(1)
    p_avg /= args.k_aug
    p_mix = args.alpha * lo_clean.softmax(1) + (1 - args.alpha) * p_avg
    target_logp = F.log_softmax(torch.log(p_mix.clamp_min(1e-8)) / args.tau, 1)  # [N,100] sur GPU
    print(f"[target] {stats(target_logp, y)}")

    # ---- modele a entrainer
    student = copy.deepcopy(orig)
    for p in student.parameters():
        p.requires_grad_(args.train_blocks == 0)
    if args.train_blocks > 0:
        for m in [*student.blocks[-args.train_blocks:], student.norm, student.head]:
            for p in m.parameters():
                p.requires_grad_(True)
    trainable = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    w0 = {n: p.detach().clone() for n, p in trainable} if args.l2sp > 0 else None
    opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(N / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def kl_log(t_logp, s_logits):  # KL(target || student), cibles en log-probas
        return F.kl_div(F.log_softmax(s_logits, 1), t_logp, reduction="batchmean", log_target=True)

    history = []
    for ep in range(args.epochs):
        student.train()
        perm = torch.randperm(N)
        for b in range(0, N, args.bs):
            idx = perm[b:b + args.bs]
            B = len(idx)
            x = to_unit(forget[idx])
            xa = augment(x, args.aug_train)
            xm = mosaic(forget, torch.randint(0, N, (B, 4)))
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lo_a = orig(norm(xa)).float()
                lo_m = orig(norm(xm)).float()
            if args.aug_target == "same":
                t_a = target_logp[idx.to(DEVICE)]     # le gold predit ~ pareil sur l'image et sa variante legere
            else:
                t_a = F.log_softmax(lo_a / args.tau, 1)
            t_m = F.log_softmax(lo_m / 2, 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = student(norm(torch.cat([x, xa, xm]))).float()
            s_x, s_a, s_m = out[:B], out[B:2 * B], out[2 * B:]
            loss = kl_log(target_logp[idx.to(DEVICE)], s_x)
            loss = loss + args.w_aug * kl_log(t_a, s_a)
            loss = loss + args.w_ret * 4.0 * kl_log(t_m, s_m / 2)
            if w0 is not None:
                loss = loss + args.l2sp * sum(((p - w0[n]) ** 2).sum() for n, p in trainable)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            sched.step()

        # ---- eval locale (proxy, pas le vrai score)
        ls_clean = logits_u8(student, forget)
        ls_aug = logits_u8(student, forget, aug=1.0, seed=1)
        ls_mos = logits_from_unit(student, mos_eval, len(mos_idx))
        rec = {
            "epoch": ep + 1,
            "forget_clean": stats(ls_clean, y),
            "forget_aug": stats(ls_aug, y),
            "kl_target_clean": kl_t2(target_logp, ls_clean),
            "kl_orig_mosaic_T2": kl_t2(lo_mos, ls_mos),
            "agree_orig_mosaic": (ls_mos.argmax(1) == lo_mos.argmax(1)).float().mean().item(),
            "min": round((time.time() - t0) / 60, 1),
        }
        history.append(rec)
        print(json.dumps(rec))

    out = Path(args.out)
    if not out.is_absolute():
        out = VDIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf dans les poids"
    torch.save(sd, out)
    out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
