#!/usr/bin/env python
"""Unlearning v36 - ancrage sur des images EXTERIEURES au lieu des mosaiques d'images a oublier.

Nouveau en v36 (le reste = v30, recette du record) :
  * --ext_anchor fichier.npy : les images d'ancrage ("reste comme l'original ici") sont des photos exterieures (COCO),
    et plus des mosaiques fabriquees avec les images a oublier. Ces mosaiques avaient un defaut : elles remontraient au
    modele, en morceaux, les images qu'il doit oublier.
  * --ext_frac F : part des images d'ancrage prises a l'exterieur (1 = toutes, 0.5 = moitie photos, moitie mosaiques).

v30 - choix explicite des classes "retirees en entier".

Nouveau en v30 (le reste = v28, record) :
  * --full_classes "0,29,30" : liste explicite des classes traitees comme retirees en entier (au lieu de "toutes celles
    qui ont >= 125 images"). Les autres classes a 130 images sont alors traitees comme des classes partielles.
    Motif : v27 montre que ne PAS eteindre le groupe {6,7,8} fait monter le score -> le gold les connait sans doute.

v10 - "inclinaison" de la mauvaise reponse vers les classes ressemblantes (idee TRW, arXiv 2512.06993).

Nouveau en v10 (le reste = v9/v5) :
  * --tilt B : pour les images des classes entieres, on ajoute B x ressemblance(classe retiree, classe k) au logit
    de chaque classe k restante dans la cible. Ressemblance = cosinus entre les vecteurs de poids de la derniere
    couche de l'original. Un modele reentraine sans une classe repond surtout les classes qui lui ressemblent.
    Les 9 classes entieres ne recoivent pas d'inclinaison (le gold ne les predit jamais).

v9 - "professeur coherent" : l'original avec les 9 classes entieres deja eteintes.

Nouveau en v9 (le reste = v5/v7) :
  * --teacher_beta B : le professeur = l'original dont on baisse de B les logits des classes entieres, PARTOUT
    (cibles du forget set, vues modifiees, mosaiques). Plus de contradiction entre "oublier" et "imiter l'original".
  * --init_shift : l'eleve demarre deja avec ces classes eteintes (biais de la derniere couche - B).
  * --k_aug K (avec --target delta) : base de la cible = moyenne des reponses du professeur sur K vues modifiees
    (le gold est une moyenne de plusieurs modeles -> cible plus "moyennee" pour le choix de la mauvaise classe).
  * --delta_slope S : oubli proportionnel a la part retiree de la classe (delta = delta_part + S x (part - part moyenne)).

v7 - "reparation du retain" apres l'extinction des 9 classes entieres (v6) :

Nouveau en v7 (le reste = v5) :
  * --init : l'eleve part d'un modele deja oublie (ex. v6/subs/f20_p8_b8.pt) au lieu de l'original.
  * --target init : cible sur le forget set = les predictions de ce modele de depart (on GARDE l'oubli).
  * --mos_exclude_full : mosaiques d'ancrage construites SANS les images des classes entieres.
    -> l'eleve redevient comme l'original sur des images qui ne sont pas de ces classes (retain),
       sans qu'on lui "rappelle" les classes eteintes.

v5 :

Nouveau en v5 (le reste = v4) :
  * Le diag v4 montre 9 classes a 129-130 images (plafond identique = classe entiere supprimee ?) et
    36 classes partielles (46-104 images). Le gold n'a jamais vu les 9 classes entieres -> ~0 % d'acc
    dessus ; il connait les classes partielles -> comportement "image de test".
  * Classe "entiere" = classe dont le nb d'images (pseudo-labels) >= --full_thr.
  * --delta_full s'applique aux images des classes entieres, --delta_part aux autres (--target delta).
  * --mos_full_delta : option, baisse aussi ces classes dans la cible des mosaiques (0 = mosaiques = original).
  * Eval par epoque separee : classes entieres / classes partielles.

Nouveau en v4 (le reste = v2) :
  * --target delta : cible = logits de l'original, MOINS delta sur le logit de la classe memorisee.
    Les autres classes gardent leur ordre (= les classes visuellement proches que le gold choisirait
    a la place). Un seul reglage, delta, controle la force d'oubli. Les images dont la 2e classe est
    plausible basculent, les autres restent justes -> distribution de loss a queue lourde naturelle.
  * --diag : en plus, histogramme des classes du forget set (images au hasard ou classes entieres ?),
    marge top1-top2 de l'original, et stats de la cible pour plusieurs delta (sans entrainement).
  * A chaque epoque : "fit" = la metrique officielle (mean_std x prediction, T=2) en prenant la CIBLE
    comme gold. Mesure si l'eleve atteint la cible (1 = parfait), pas le vrai score.

Idee de base (v1/v2) :
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


def delta_target(base_logits: torch.Tensor, y: torch.Tensor, delta: float) -> torch.Tensor:
    """Retire `delta` au logit de la classe memorisee y. Renvoie des logits."""
    t = base_logits.clone()
    t[torch.arange(len(y), device=t.device), y] -= delta
    return t


def metric_vs(gold, sub, orig, y) -> dict:
    """Metrique officielle (softmax T=2) avec `gold` donne. Ici gold = notre cible -> mesure du fit."""
    def L(lo):
        return F.cross_entropy(lo / 2, y, reduction="none")

    def d(lo):
        return (L(lo).mean() - L(gold).mean()).abs() + (L(lo).std() - L(gold).std()).abs()

    def kl(lo):
        lp, lq = F.log_softmax(gold / 2, 1), F.log_softmax(lo / 2, 1)
        return (lp.exp() * (lp - lq)).sum(1).mean()

    ms = 1 - (d(sub) / d(orig).clamp_min(1e-8)).clamp(0, 1)
    pr = 1 - (kl(sub) / kl(orig).clamp_min(1e-8)).clamp(0, 1)
    return {"mean_std": ms.item(), "pred": pr.item(), "fit": (ms * pr).item()}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="stats du modele original uniquement")
    ap.add_argument("--init", type=str, default=None, help="poids de depart de l'eleve (defaut = original)")
    ap.add_argument("--mos_exclude_full", action="store_true", help="mosaiques sans les images des classes entieres")
    ap.add_argument("--target", choices=["avg", "delta", "init"], default="avg",
                    help="avg = moyenne des vues augmentees (v1-v3) ; delta = logit memorise baisse de delta (v4)")
    ap.add_argument("--delta", type=float, default=8.0, help="(target delta) baisse du logit de la classe memorisee")
    ap.add_argument("--full_classes", type=str, default=None, help="liste explicite des classes entieres, ex. 0,29,30")
    ap.add_argument("--ext_anchor", type=str, default=None, help="fichier .npy d'images exterieures pour l'ancrage")
    ap.add_argument("--ext_frac", type=float, default=1.0)
    ap.add_argument("--ext_aug", type=float, default=1.0, help="v57 : force des modifications appliquees aux images d'ancrage (liste de fichiers .npy separes par des virgules dans --ext_anchor)")
    ap.add_argument("--tilt", type=float, default=0.0, help="force de l'inclinaison vers les classes ressemblantes (classes entieres)")
    ap.add_argument("--teacher_beta", type=float, default=0.0, help="baisse des logits des classes entieres chez le professeur")
    ap.add_argument("--init_shift", action="store_true", help="l'eleve demarre avec les classes entieres eteintes (biais - teacher_beta)")
    ap.add_argument("--delta_slope", type=float, default=0.0, help="delta partiel += slope x (part retiree de la classe - moyenne)")
    ap.add_argument("--class_size", type=float, default=130.0, help="taille supposee d'une classe complete")
    ap.add_argument("--teacher_cache", type=str, default=None, help="v47 : noms de caches v45/cache_m_<nom>.pt de modeles de-memorises, separes par des virgules")
    ap.add_argument("--tail_mix", type=float, default=1.0, help="v47 : 1 = hesitations propres a chaque image ; < 1 = melangees (a T=2) avec les hesitations MOYENNES des images de la meme classe")
    ap.add_argument("--teacher_T", type=float, default=1.0, help="v47 : temperature a laquelle on moyenne les estimations (2 = comme le gold)")
    ap.add_argument("--extra_cache", type=str, default=None); ap.add_argument("--extra_mix", type=float, default=0.0)
    ap.add_argument("--teacher_mix", type=float, default=0.5, help="v47 : part des professeurs dans la cible (moyenne de probabilites, comme le gold qui est une moyenne de modeles)")
    ap.add_argument("--part_floor", type=float, default=None, help="v47 : pour les classes PARTIELLES, la bonne classe reste au moins part_floor au-dessus de la meilleure autre dans la cible (on ne fait plus basculer leurs images fragiles)")
    ap.add_argument("--sac_frac", type=float, default=0.0, help="v47 : part des images 'sacrifiees' (loss tres haute), choisies parmi les plus fragiles")
    ap.add_argument("--sac_scope", choices=["all", "heavy"], default="all", help="v47 : parmi toutes les images, ou seulement les 9 classes a 130 images")
    ap.add_argument("--sac_margin", type=float, default=16.0, help="v47 : loss visee sur les images sacrifiees")
    ap.add_argument("--w_sac", type=float, default=0.3)
    ap.add_argument("--sib_k", type=int, default=0, help="v46 : nb de classes voisines remontees dans la cible")
    ap.add_argument("--sib_delta", type=float, default=0.0)
    ap.add_argument("--sib_full", type=float, default=0.0)
    ap.add_argument("--mid_classes", type=str, default=None, help="v46 : classes au delta intermediaire")
    ap.add_argument("--delta_mid", type=float, default=12.0)
    ap.add_argument("--delta_cls", type=str, default=None, help="v46 : delta par classe, ex. 5:4,32:8")
    ap.add_argument("--full_thr", type=int, default=125, help="classe 'entiere' si nb d'images >= full_thr")
    ap.add_argument("--delta_full", type=float, default=None, help="delta pour les classes entieres (defaut = --delta)")
    ap.add_argument("--delta_part", type=float, default=None, help="delta pour les classes partielles (defaut = --delta)")
    ap.add_argument("--mos_full_delta", type=float, default=0.0,
                    help="baisse des logits des classes entieres dans la cible des mosaiques (0 = desactive)")
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
        # 1) repartition par classe : images au hasard (~35/classe) ou classes entieres ?
        cnt = torch.bincount(y, minlength=lo_clean.shape[1]).cpu()
        nz = cnt[cnt > 0].float()
        print(f"[diag] classes presentes {int((cnt > 0).sum())}/100 | par classe : min {int(nz.min())} "
              f"mediane {int(nz.median())} max {int(nz.max())} | ecart-type {nz.std().item():.1f}")
        top = cnt.argsort(descending=True)[:15]
        print("[diag] top classes (classe:nb) " + " ".join(f"{int(c)}:{int(cnt[c])}" for c in top))
        print("[diag] histogramme complet " + json.dumps(cnt.tolist()))
        # 2) marge de memorisation : logit top1 - top2
        s = lo_clean.sort(1, descending=True).values
        m = (s[:, 0] - s[:, 1]).cpu()
        q = torch.quantile(m, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        print(f"[diag] marge top1-top2 : moyenne {m.mean():.2f} | quantiles 10/25/50/75/90 % "
              + " ".join(f"{v:.2f}" for v in q.tolist()))
        # 3) stats de la cible delta (T=1 et T=2) pour plusieurs delta, sans entrainement
        for dl in (2, 4, 6, 8, 10, 12, 14, 16, 20):
            t = delta_target(lo_clean, y, dl)
            st, st2 = stats(t, y), stats(t / 2, y)
            print(f"[diag] delta={dl:<3} T1 {json.dumps({k: round(v, 3) for k, v in st.items()})} "
                  f"| T2 loss {st2['loss_mean']:.3f} +- {st2['loss_std']:.3f}")
        return

    # ---- classes supprimees en entier (v5)
    cnt = torch.bincount(y, minlength=lo_clean.shape[1])
    full_cls = (cnt >= args.full_thr).nonzero().flatten()
    if args.full_classes is not None:
        full_cls = torch.tensor([int(c) for c in args.full_classes.split(",") if c != ""], device=y.device, dtype=torch.long)
    full = torch.isin(y, full_cls)                     # [N] bool : image d'une classe entiere ?
    print(f"[v5] classes entieres (>= {args.full_thr} images) : {full_cls.tolist()} -> {int(full.sum())} images, "
          f"{int((~full).sum())} images de classes partielles")

    # ---- construction de la cible "non memorisee"
    if args.target == "delta":
        if args.k_aug > 0:   # base = moyenne des logits sur k vues augmentees, sinon image exacte
            base = sum(logits_u8(orig, forget, aug=args.aug, seed=1000 + k) for k in range(args.k_aug)) / args.k_aug
        else:
            base = lo_clean
        base = base.clone()
        base[:, full_cls] -= args.teacher_beta            # v9 : le professeur a les classes entieres eteintes
        d_full = args.delta if args.delta_full is None else args.delta_full
        d_part = args.delta if args.delta_part is None else args.delta_part
        frac = cnt[y].float() / args.class_size            # part de la classe retiree (0.35 a 0.8 pour les partielles)
        d_img = d_part + args.delta_slope * (frac - frac[~full].mean())
        dvec = torch.where(full, torch.full_like(d_img, d_full), d_img)
        # v46 : classes "intermediaires" (ex. lanternes, bateaux) avec leur propre delta ; --delta_cls "5:4,32:8" = delta par classe
        if args.mid_classes:
            mid_cls = torch.tensor([int(c) for c in args.mid_classes.split(",") if c != ""], device=y.device, dtype=torch.long)
            mid = torch.isin(y, mid_cls) & ~full; dvec = torch.where(mid, torch.full_like(dvec, args.delta_mid), dvec)
            print(f"[v46] classes intermediaires {mid_cls.tolist()} : {int(mid.sum())} images, delta {args.delta_mid}")
        if args.delta_cls:
            for item in args.delta_cls.split(","):
                c_, v_ = item.split(":"); dvec = torch.where((y == int(c_)) & ~full, torch.full_like(dvec, float(v_)), dvec)
            print(f"[v46] delta par classe : {args.delta_cls}")
        if args.part_floor is not None:
            ar_ = torch.arange(len(y), device=y.device); o_ = base.clone(); o_[ar_, y] = -1e9; m_ = base[ar_, y] - o_.max(1).values
            part_ = cnt[y] < args.full_thr; d_new = torch.minimum(dvec, (m_ - args.part_floor).clamp_min(0.0))
            print(f"[v47] classes partielles : {int((part_ & (d_new < dvec)).sum())} images fragiles gardees justes (delta reduit)")
            dvec = torch.where(part_, d_new, dvec)
        print(f"[v9] delta partiel par image : min {d_img[~full].min():.2f} moy {d_img[~full].mean():.2f} max {d_img[~full].max():.2f}")
        tgt = delta_target(base, y, dvec)
        if args.sib_k > 0:
            # v46 : oubli "cible sur les voisines" : au lieu de baisser la vraie classe face a TOUTES les autres (ce qui gonfle
            # la probabilite des 96 classes sans rapport), on remonte seulement les sib_k classes les plus proches de l'image
            # (d'apres la moyenne sur les vues modifiees). Les classes retirees (0, 38) ne sont jamais des voisines.
            tmp = base.clone(); tmp[torch.arange(len(y), device=y.device), y] = -1e9; tmp[:, full_cls] = -1e9
            top = tmp.topk(args.sib_k, 1).indices
            add = torch.zeros_like(tgt).scatter_(1, top, 1.0)
            tgt = tgt + add * torch.where(full, torch.full_like(dvec, args.sib_full), torch.full_like(dvec, args.sib_delta))[:, None]
            print(f"[v46] voisines : {args.sib_k} classes remontees de {args.sib_delta} (images ordinaires) / {args.sib_full} (classes retirees)")
        if args.tilt > 0:                                  # v10 : inclinaison vers les classes ressemblantes
            Wn = F.normalize(orig.head.weight.detach().float(), dim=1)
            sim = Wn @ Wn.T                                # [100,100] cosinus entre classes
            sim[:, full_cls] = 0.0                         # pas d'inclinaison vers une classe entiere
            tgt[full] += args.tilt * sim[y[full]]
            top = sim[full_cls].topk(3, dim=1)
            for c, v, i in zip(full_cls.tolist(), top.values.tolist(), top.indices.tolist()):
                print(f"[v10] classe {c} : plus ressemblantes {i} (cos {[round(x, 2) for x in v]})")
        sac = torch.zeros(len(y), dtype=torch.bool, device=y.device)
        if args.sac_frac > 0:
            # v47 : le gold se trompe "avec assurance" sur une petite part des images (loss enorme). Plutot que d'eteindre des
            # classes entieres, on choisit les images les plus FRAGILES (petite marge de la vraie classe sur les vues modifiees) :
            # ce sont celles qu'un modele qui ne les a jamais vues a le plus de chances de rater. Leur cible = classe voisine.
            ar = torch.arange(len(y), device=y.device)
            oth = base.clone(); oth[ar, y] = -1e9; marg = base[ar, y] - oth.max(1).values
            pool = torch.ones_like(sac) if args.sac_scope == "all" else torch.isin(y, (cnt >= args.full_thr).nonzero().flatten())
            score = marg.clone(); score[~pool] = 1e9
            k_sac = int(round(args.sac_frac * len(y))); sac[score.topk(k_sac, largest=False).indices] = True
            tgt[sac, y[sac]] = base[sac, y[sac]] - 40.0
            _o = Path(args.out); _o = _o if _o.is_absolute() else VDIR / _o; _o.parent.mkdir(parents=True, exist_ok=True); torch.save({"sac": sac.cpu(), "marge": marg.cpu()}, str(_o.with_suffix("")) + "_sac.pt")
            print(f"[v47] {int(sac.sum())} images sacrifiees (marge <= {marg[sac].max():.2f}) ; par classe : " + " ".join(f"{int(c_)}:{int(v_)}" for v_, c_ in zip(*torch.bincount(y[sac], minlength=100).topk(12))))
        if args.teacher_cache:
            # v47 : cible "d'ensemble" = moyenne (en probabilites) de deux estimations independantes de ce que repondrait un modele
            # qui n'a pas vu l'image : (1) l'original sur des vues modifiees (notre cible habituelle), (2) des modeles de-memorises.
            Pt = 0; tags = [t_ for t_ in args.teacher_cache.split(",") if t_]; TT = args.teacher_T
            for t_ in tags: Pt = Pt + (torch.load(ROOT / "v45" / f"cache_m_{t_}.pt")["e"].float().to(y.device) / TT).softmax(1)
            Pt = Pt / len(tags); mixw = args.teacher_mix
            if args.extra_cache:                       # autres estimations (ex. modele elague) : cle "nom" dans v45/cache_fragile2.pt
                f2 = torch.load(ROOT / "v45" / "cache_fragile2.pt"); Pe = 0; ex = [e_ for e_ in args.extra_cache.split(",") if e_]
                for e_ in ex: Pe = Pe + (f2[e_].float().to(y.device) / TT).softmax(1)
                Pt = (mixw * Pt + args.extra_mix * Pe / len(ex)) / (mixw + args.extra_mix); mixw = mixw + args.extra_mix
            Pv = F.softmax(tgt / args.tau / TT, 1); keep = ~sac                      # les images sacrifiees gardent leur cible
            Pm = (1 - mixw) * Pv + mixw * Pt                                          # moyenne faite a la temperature TT (le gold moyenne a T=2)
            tgt = torch.where(keep[:, None], torch.log(Pm.clamp_min(1e-12)) * args.tau * TT, tgt)
            print(f"[v47] cible d'ensemble : {len(tags)} professeurs, part {args.teacher_mix} ; accord (meme 1er choix) vues/professeurs : {(Pv.argmax(1) == Pt.argmax(1)).float().mean():.3f}")
        if args.tail_mix < 1.0:
            # v47 : lissage des "mauvaises classes" a T=2 : on garde la probabilite de la vraie classe de chaque image, mais la
            # repartition du reste est un melange entre celle de l'image et la moyenne de sa classe (moins de bruit image par image).
            ar_ = torch.arange(len(y), device=y.device); P2 = F.softmax(tgt / args.tau / 2, 1); pt2 = P2[ar_, y].clone()
            Q = P2.clone(); Q[ar_, y] = 0; Q = Q / Q.sum(1, keepdim=True); Qc = torch.zeros(100, 100, device=y.device); ns_ = ~sac
            for c_ in y.unique().tolist():
                m_ = (y == c_) & ns_
                if m_.any(): Qc[c_] = Q[m_].mean(0)
            Qm = args.tail_mix * Q + (1 - args.tail_mix) * Qc[y]; Qm[ar_, y] = 0; Qm = Qm / Qm.sum(1, keepdim=True)
            Pn = Qm * (1 - pt2)[:, None]; Pn[ar_, y] = pt2
            tgt = torch.where(ns_[:, None], torch.log(Pn.clamp_min(1e-12)) * args.tau * 2, tgt)
            print(f"[v47] hesitations lissees a T=2 : {args.tail_mix} image / {1 - args.tail_mix:.2f} moyenne de la classe")
        target_logp = F.log_softmax(tgt / args.tau, 1)
        print(f"[target] classes entieres (delta {d_full}) {stats(target_logp[full], y[full])}")
        print(f"[target] classes partielles (delta {d_part}) {stats(target_logp[~full], y[~full])}")
    elif args.target == "init":   # v7 : garder l'oubli du modele de depart
        assert args.init, "--target init demande --init"
        init_model = load_model(Path(args.init).resolve())
        target_logp = F.log_softmax(logits_u8(init_model, forget) / args.tau, 1)
        del init_model
        print(f"[target] classes entieres {stats(target_logp[full], y[full])}")
        print(f"[target] classes partielles {stats(target_logp[~full], y[~full])}")
    else:
        p_avg = torch.zeros_like(lo_clean)
        for k in range(args.k_aug):
            p_avg += logits_u8(orig, forget, aug=args.aug, seed=1000 + k).softmax(1)
        p_avg /= args.k_aug
        p_mix = args.alpha * lo_clean.softmax(1) + (1 - args.alpha) * p_avg
        target_logp = F.log_softmax(torch.log(p_mix.clamp_min(1e-8)) / args.tau, 1)  # [N,100] sur GPU
    print(f"[target] {stats(target_logp, y)}")

    # ---- modele a entrainer
    student = load_model(Path(args.init).resolve()) if args.init else copy.deepcopy(orig)
    if args.init_shift:
        with torch.no_grad():
            student.head.bias[full_cls] -= args.teacher_beta
    mos_pool = (~full).nonzero().flatten().cpu() if args.mos_exclude_full else torch.arange(N)
    print(f"[v7] init = {args.init or 'original'} | mosaiques sur {len(mos_pool)} images")
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

    EXT = None
    if args.ext_anchor:
        import numpy as _np
        EXT = torch.cat([torch.from_numpy(_np.load(p_)) for p_ in args.ext_anchor.split(",")]); print(f"[v57] ancrage sur {len(EXT)} images (fichiers : {args.ext_anchor} ; part {args.ext_frac} ; force des modifications {args.ext_aug})")
    history = []
    for ep in range(args.epochs):
        torch.manual_seed(args.seed * 1000 + ep)   # v28 : le hasard de l'entrainement depend VRAIMENT de --seed
        student.train()
        perm = torch.randperm(N)
        for b in range(0, N, args.bs):
            idx = perm[b:b + args.bs]
            B = len(idx)
            x = to_unit(forget[idx])
            xa = augment(x, args.aug_train)
            xm = mosaic(forget, mos_pool[torch.randint(0, len(mos_pool), (B, 4))])
            if EXT is not None:
                ne = int(round(args.ext_frac * B))
                if ne > 0: xm = torch.cat([augment(to_unit(EXT[torch.randint(0, len(EXT), (ne,))]), args.ext_aug), xm[ne:]])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lo_a = orig(norm(xa)).float()
                lo_m = orig(norm(xm)).float()
            if args.aug_target == "same":
                t_a = target_logp[idx.to(DEVICE)]     # le gold predit ~ pareil sur l'image et sa variante legere
            else:
                t_a = F.log_softmax(lo_a / args.tau, 1)
            if args.teacher_beta > 0 and len(full_cls):
                lo_m[:, full_cls] -= args.teacher_beta      # v9 : mosaiques = professeur coherent
            if args.mos_full_delta > 0 and len(full_cls):
                lo_m[:, full_cls] -= args.mos_full_delta
            t_m = F.log_softmax(lo_m / 2, 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = student(norm(torch.cat([x, xa, xm]))).float()
            s_x, s_a, s_m = out[:B], out[B:2 * B], out[2 * B:]
            loss = kl_log(target_logp[idx.to(DEVICE)], s_x)
            loss = loss + args.w_aug * kl_log(t_a, s_a)
            loss = loss + args.w_ret * 4.0 * kl_log(t_m, s_m / 2)
            if args.sac_frac > 0:
                ib = idx.to(DEVICE); sb = sac[ib]
                if sb.any():
                    yb = y[ib][sb]; ar_b = torch.arange(len(yb), device=DEVICE)
                    for s_ in (s_x, s_a):                    # image exacte et copie legere : la vraie classe doit tomber a "loss = sac_margin"
                        zs = s_[sb]; zo = zs.clone(); zo[ar_b, yb] = -1e9
                        viol = F.relu(zs[ar_b, yb] - torch.logsumexp(zo, 1) + args.sac_margin)
                        loss = loss + args.w_sac * F.smooth_l1_loss(viol, torch.zeros_like(viol), reduction="sum") / B
            if w0 is not None:
                loss = loss + args.l2sp * sum(((p - w0[n]) ** 2).sum() for n, p in trainable)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            sched.step()

        # ---- eval locale (proxy, pas le vrai score)
        ls_clean = logits_u8(student, forget)
        if args.sac_frac > 0: print(f"[v47] epoque {ep + 1} : loss images sacrifiees {F.cross_entropy(ls_clean[sac], y[sac], reduction='none').mean():.2f} +- {F.cross_entropy(ls_clean[sac], y[sac], reduction='none').std():.2f} | autres {F.cross_entropy(ls_clean[~sac], y[~sac]).item():.3f}", flush=True)
        ls_aug = logits_u8(student, forget, aug=1.0, seed=1)
        ls_mos = logits_from_unit(student, mos_eval, len(mos_idx))
        rec = {
            "epoch": ep + 1,
            "forget_clean": stats(ls_clean, y),
            "forget_aug": stats(ls_aug, y),
            "kl_target_clean": kl_t2(target_logp, ls_clean),
            "fit_vs_target": metric_vs(target_logp, ls_clean, lo_clean, y),
            "clean_full": stats(ls_clean[full], y[full]),
            "clean_part": stats(ls_clean[~full], y[~full]),
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
