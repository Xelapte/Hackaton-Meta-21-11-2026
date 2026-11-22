#!/usr/bin/env python
"""v24 - QUELLE TRANSFORMATION "DEMEMORISE" LE MIEUX ? (diagnostic pur, dans la maquette v21)

Notre cible vient des reponses de l'original sur des vues modifiees de l'image. Mais l'original reconnait encore par
coeur les vues legerement modifiees. Question : quelle transformation (zoom fort, basse resolution, flou, rotation,
gris, effacement, melange de morceaux...) fait repondre l'original COMME UN GOLD, c'est-a-dire comme sur une image
jamais vue ? On mesure, pour chaque transformation, l'ecart au vrai gold (KL, T=2) sur les classes partielles.
Aucun entrainement d'eleve : juste des mesures.

Maquette REALISTE de l'epreuve (suite de v20).

Defaut de v20 : un petit reseau entraine de zero generalise mal (33 % sur le controle) -> regime tres different du vrai.
Ici : on garde les 8 premiers blocs du VRAI modele (figes : ils savent "voir") et on re-entraine DE ZERO les 4 derniers
blocs + la tete, sur nos 3500 images (45 classes). On obtient un "original maquette" qui generalise bien et qui apprend
par coeur ses images d'entrainement, et de vrais "gold maquette" (3 graines) reentraines sans le forget set maquette
(4 classes entieres + la moitie de ~20 classes). Puis formule OFFICIELLE du score, avec un vrai gold.
Limite : les 8 premiers blocs ont vu toutes les images pendant le vrai entrainement.
"""
import sys, json, time, copy, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v24"
sys.path.insert(0, str(B / "v19")); sys.path.insert(0, str(B))
import unlearn as U  # noqa: E402  (augment ; fait chdir vers unlearning/)
from task_template import build_model, load_model  # noqa: E402
dev = "cuda"; t0 = time.time(); NB = 8; NV = 6
def log(*a): print(f"[{(time.time() - t0) / 60:5.1f} min]", *a, flush=True)

g = torch.Generator().manual_seed(0)
IMG = torch.from_numpy(np.load(B / "forget_images.npz")["images"]).to(dev)          # uint8 [N,224,224,3]
y_real = torch.load(B / "v8" / "cache" / "_meta.pt")["y"]
classes = y_real.unique().tolist(); K = len(classes); N = len(IMG)
y = torch.tensor([classes.index(int(c)) for c in y_real], device=dev)
FULL_REAL = [2, 5, 32, 36]
full_cls = torch.tensor([classes.index(c) for c in FULL_REAL], device=dev)
others = [k for k in range(K) if k not in full_cls.tolist()]; part_cls = others[::2]
H, T, Fo = [], [], []
for k in range(K):
    idx = (y == k).nonzero().flatten().cpu(); idx = idx[torch.randperm(len(idx), generator=g)]
    nh = max(1, round(0.2 * len(idx))); H += idx[:nh].tolist(); tr = idx[nh:].tolist(); T += tr
    if k in full_cls.tolist(): Fo += tr
    elif k in part_cls: Fo += tr[: len(tr) // 2]
H, T, Fo = torch.tensor(H), torch.tensor(T), torch.tensor(Fo)
R = torch.tensor(sorted(set(T.tolist()) - set(Fo.tolist()))); is_full = torch.isin(y, full_cls)
log(f"{K} classes | controle {len(H)} | entrainement {len(T)} | forget {len(Fo)} (dont {int(is_full[Fo].sum())} de classes entieres) | retain {len(R)}")

real = load_model(B / "model.pt")
for p in real.parameters(): p.requires_grad_(False)
def unit(idx): return IMG[idx.to(dev)].permute(0, 3, 1, 2).float() / 255
def norm(x): return (x - 0.5) / 0.5
@torch.no_grad()
def lower(x):                                   # images [n,3,224,224] dans [0,1] -> jetons apres le bloc NB (blocs figes du vrai modele)
    out = []
    for b in range(0, len(x), 250):
        with torch.autocast("cuda", dtype=torch.float16):
            t = real.patch_embed(norm(x[b:b + 250])); t = real._pos_embed(t); t = real.patch_drop(t); t = real.norm_pre(t)
            for blk in real.blocks[:NB]: t = blk(t)
        out.append(t.half())
    return torch.cat(out)
class Top(nn.Module):                           # 4 derniers blocs + norme + tete, initialises AU HASARD
    def __init__(self, seed):
        super().__init__(); torch.manual_seed(seed); m = build_model()
        self.blocks = nn.Sequential(*m.blocks[NB:]); self.norm = m.norm; self.head = nn.Linear(768, K)
    def forward(self, t): return self.head(self.norm(self.blocks(t))[:, 0])
allidx = torch.arange(N)
torch.manual_seed(11)
TOK = [torch.cat([lower(unit(allidx[i:i + 500]) if v == 0 else U.augment(unit(allidx[i:i + 500]), 0.6)) for i in range(0, N, 500)]) for v in range(NV)]
log(f"jetons en cache : {NV} vues")

def train(idx, seed, epochs=80, lr=5e-4):
    m = Top(seed).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(idx) / 128); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    torch.manual_seed(seed + 100)
    for ep in range(epochs):
        m.train(); perm = idx[torch.randperm(len(idx))]
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128].to(dev); v = int(torch.randint(1, NV, (1,)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(m(TOK[v][i]).float(), y[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return m.eval()

@torch.no_grad()
def lg(m, tok, shift=0.0):
    m.eval(); out = []
    for b in range(0, len(tok), 500):
        with torch.autocast("cuda", dtype=torch.float16): out.append(m(tok[b:b + 500]).float())
    z = torch.cat(out)
    if shift: z = z.clone(); z[:, full_cls] -= shift
    return z
torch.manual_seed(77); TOK_ALT = lower(U.augment(unit(Fo), 0.5))                 # "copies modifiees" du forget set (fixes)
TOK_F, TOK_H = TOK[0][Fo.to(dev)], TOK[0][H.to(dev)]
def views(m, shift=0.0): return {"c": lg(m, TOK_F, shift), "a": lg(m, TOK_ALT, shift), "h": lg(m, TOK_H, shift)}

orig = train(T, 0); log("original maquette entraine")
golds = [train(R, s) for s in (1, 2, 3)]; log("3 gold maquette entraines")
yF, yH = y[Fo.to(dev)], y[H.to(dev)]; fF = is_full[Fo.to(dev)]; hNF = ~is_full[H.to(dev)]
vo = views(orig); vg = [views(m) for m in golds]
ce = lambda z, yy: F.cross_entropy(z, yy, reduction="none")
GOLD = {k: {"p": torch.stack([F.softmax(v[k] / 2, 1) for v in vg]).mean(0), "mean": torch.stack([ce(v[k], yF).mean() for v in vg]).mean(),
            "std": torch.stack([ce(v[k], yF).std() for v in vg]).mean()} for k in ("c", "a")}

def score(v):
    r = {}
    for k in ("c", "a"):
        G = GOLD[k]
        d = lambda z: (ce(z, yF).mean() - G["mean"]).abs() + (ce(z, yF).std() - G["std"]).abs()
        klv = lambda z: (G["p"] * (G["p"].clamp_min(1e-9).log() - F.log_softmax(z / 2, 1))).sum(1)
        r["ms_" + k] = (1 - (d(v[k]) / d(vo[k])).clamp(0, 1)).item(); r["pr_" + k] = (1 - (klv(v[k]).mean() / klv(vo[k]).mean()).clamp(0, 1)).item()
        if k == "c":
            r["kl_full"] = klv(v[k])[fF].mean().item(); r["kl_part"] = klv(v[k])[~fF].mean().item()
            r["L_full"] = ce(v[k], yF)[fF].mean().item(); r["L_part"] = ce(v[k], yF)[~fF].mean().item(); r["L_std"] = ce(v[k], yF).std().item()
    r["forget"] = (r["ms_c"] + r["ms_a"]) / 2 * (r["pr_c"] + r["pr_a"]) / 2
    po = F.log_softmax(vo["h"] / 2, 1); ps = F.log_softmax(v["h"] / 2, 1)
    for name, msk in (("ret_all", torch.ones_like(hNF)), ("ret_nofull", hNF)):
        acc_s = (v["h"][msk].argmax(1) == yH[msk]).float().mean(); acc_o = (vo["h"][msk].argmax(1) == yH[msk]).float().mean()
        r[name] = ((acc_s / acc_o).clamp(0, 1) / (1 + (po[msk].exp() * (po[msk] - ps[msk])).sum(1).mean())).item()
    r["score_all"] = r["forget"] * r["ret_all"]; r["score_nofull"] = r["forget"] * r["ret_nofull"]
    r["acc_full"] = (v["c"][fF].argmax(1) == yF[fF]).float().mean().item(); r["acc_part"] = (v["c"][~fF].argmax(1) == yF[~fF]).float().mean().item()
    r["acc_ctrl"] = (v["h"].argmax(1) == yH).float().mean().item()
    return r
RES = {}
def rep(name, v):
    r = score(v); RES[name] = r
    log(f"{name:40} SCORE {r['score_all']:.3f}/{r['score_nofull']:.3f} | oubli {r['forget']:.2f} = ms {r['ms_c']:.2f}/{r['ms_a']:.2f} x pred {r['pr_c']:.2f}/{r['pr_a']:.2f} "
        f"| retain {r['ret_all']:.2f}/{r['ret_nofull']:.2f} | acc ent {r['acc_full']:.2f} part {r['acc_part']:.2f} ctrl {r['acc_ctrl']:.2f} "
        f"| L ent {r['L_full']:.1f} part {r['L_part']:.2f} std {r['L_std']:.2f} | KL ent {r['kl_full']:.2f} part {r['kl_part']:.2f}")

log("=== DIAGNOSTIC : original vs gold (maquette realiste)")
log(f"original : acc forget {(vo['c'].argmax(1) == yF).float().mean():.3f} loss {ce(vo['c'], yF).mean():.4f} | acc copies modifiees {(vo['a'].argmax(1) == yF).float().mean():.3f} | acc controle {(vo['h'].argmax(1) == yH).float().mean():.3f}")
for j, v in enumerate(vg):
    Lc = ce(v["c"], yF); p = F.softmax(v["c"], 1)
    log(f"gold {j} : acc entieres {(v['c'][fF].argmax(1) == yF[fF]).float().mean():.3f} partielles {(v['c'][~fF].argmax(1) == yF[~fF]).float().mean():.3f} | "
        f"loss entieres {Lc[fF].mean():.2f}+-{Lc[fF].std():.2f} partielles {Lc[~fF].mean():.2f}+-{Lc[~fF].std():.2f} | maxprob entieres {p[fF].max(1).values.mean():.2f} "
        f"partielles {p[~fF].max(1).values.mean():.2f} | acc controle {(v['h'].argmax(1) == yH).float().mean():.3f}")
log(f"gold : loss moyenne {GOLD['c']['mean']:.2f} +- {GOLD['c']['std']:.2f} (copies : {GOLD['a']['mean']:.2f} +- {GOLD['a']['std']:.2f}) | "
    f"accord des 3 gold entre eux sur le forget set : {((vg[0]['c'].argmax(1) == vg[1]['c'].argmax(1)) & (vg[1]['c'].argmax(1) == vg[2]['c'].argmax(1))).float().mean():.2f}")
pc = torch.tensor([float((yF[~fF] == c).sum()) for c in range(K)], device=dev)
accg = torch.stack([(v["c"].argmax(1) == yF).float() for v in vg]).mean(0)
log("gold, classes partielles : acc par classe (min/mediane/max) " + " ".join(f"{x:.2f}" for x in torch.tensor([accg[(yF == c) & ~fF].mean().item() for c in part_cls]).quantile(torch.tensor([0.0, 0.5, 1.0])).tolist()))
rep("original (reference = 0)", vo); rep("un gold seul vs la moyenne des 3", vg[0])

def mosaic(idx4):
    im = F.interpolate(unit(idx4.flatten()), size=112, mode="bilinear", antialias=True).view(-1, 4, 3, 112, 112)
    return torch.cat([torch.cat([im[:, 0], im[:, 1]], -1), torch.cat([im[:, 2], im[:, 3]], -1)], -2)
@torch.no_grad()
def orig_views(k, strength=1.0):
    out = []
    for j in range(k):
        torch.manual_seed(1000 + j); out.append(lg(orig, lower(U.augment(unit(Fo), strength))))
    return out

n = len(Fo); G = GOLD["c"]; mask = 20.0 * F.one_hot(full_cls, K).sum(0)
def rot(x, deg):
    a = (torch.rand(len(x), device=dev) * 2 - 1) * deg * math.pi / 180; th = torch.zeros(len(x), 2, 3, device=dev)
    th[:, 0, 0] = a.cos(); th[:, 0, 1] = -a.sin(); th[:, 1, 0] = a.sin(); th[:, 1, 1] = a.cos()
    return F.grid_sample(x, F.affine_grid(th, list(x.shape), align_corners=False), padding_mode="reflection", align_corners=False)
def lowres(x, r): return F.interpolate(F.interpolate(x, size=r, mode="bilinear", antialias=True), size=224, mode="bilinear")
def blur(x, k): return F.avg_pool2d(F.pad(x, [k // 2] * 4, mode="reflect"), k, 1)
def gray(x): return x.mean(1, keepdim=True).expand_as(x)
def erase(x, frac):
    x = x.clone(); s = int(224 * math.sqrt(frac))
    for i in range(len(x)):
        t = int(torch.randint(0, 224 - s + 1, (1,))); l = int(torch.randint(0, 224 - s + 1, (1,))); x[i, :, t:t + s, l:l + s] = 0.5
    return x
def shuffle(x, gsz):
    p = 224 // gsz; t = x.unfold(2, p, p).unfold(3, p, p).reshape(len(x), 3, gsz * gsz, p, p)
    t = t[:, :, torch.randperm(gsz * gsz, device=dev)].reshape(len(x), 3, gsz, gsz, p, p).permute(0, 1, 2, 4, 3, 5).reshape(len(x), 3, 224, 224)
    return t
def mixo(x, w): return w * x + (1 - w) * x[torch.randperm(len(x), device=dev)]
TR = {"aucune (image exacte)": (lambda x: x, 1), "vues 0.5": (lambda x: U.augment(x, 0.5), 8), "vues 1.0 (record)": (lambda x: U.augment(x, 1.0), 16), "vues 1.5": (lambda x: U.augment(x, 1.5), 16),
      "basse resolution 112": (lambda x: lowres(x, 112), 1), "basse resolution 64": (lambda x: lowres(x, 64), 1), "basse resolution 40": (lambda x: lowres(x, 40), 1),
      "flou 9": (lambda x: blur(x, 9), 1), "flou 17": (lambda x: blur(x, 17), 1), "gris": (gray, 1), "rotation 20": (lambda x: rot(x, 20), 8), "rotation 45": (lambda x: rot(x, 45), 8),
      "effacement 30 %": (lambda x: erase(x, 0.3), 8), "effacement 50 %": (lambda x: erase(x, 0.5), 8), "morceaux melanges 2x2": (lambda x: shuffle(x, 2), 8),
      "morceaux melanges 4x4": (lambda x: shuffle(x, 4), 8), "fondu 70 % avec une autre image": (lambda x: mixo(x, 0.7), 8), "fondu 55 %": (lambda x: mixo(x, 0.55), 8),
      "vues 1.0 + basse resolution 64": (lambda x: lowres(U.augment(x, 1.0), 64), 16), "vues 1.0 + effacement 30 %": (lambda x: erase(U.augment(x, 1.0), 0.3), 16),
      "vues 1.0 + flou 9": (lambda x: blur(U.augment(x, 1.0), 9), 16), "vues 1.0 + gris": (lambda x: gray(U.augment(x, 1.0)), 16), "vues 1.5 + basse resolution 64": (lambda x: lowres(U.augment(x, 1.5), 64), 16)}
def q(z):
    kl = (G["p"] * (G["p"].clamp_min(1e-9).log() - F.log_softmax(z / 2, 1))).sum(1); L = ce(z, yF)
    return kl[fF].mean().item(), kl[~fF].mean().item(), (z[~fF].argmax(1) == yF[~fF]).float().mean().item(), (z[~fF].argmax(1) == vg[0]["c"][~fF].argmax(1)).float().mean().item(), L[~fF].mean().item(), L[~fF].std().item()
log(f"reference : gold 1 vs moyenne -> {q(vg[1]['c'])}")
log("colonnes : KL entieres | KL partielles | acc partielles | meme reponse que gold 0 | loss partielles moy +- std   (gold : acc 0.38-0.40, loss 3.05 +- 3.2)")
OUT = {}
for name, (fn, k) in TR.items():
    zs = []
    for j in range(k):
        torch.manual_seed(3000 + j); zs.append(lg(orig, lower(fn(unit(Fo)))))
    zl = torch.stack(zs).mean(0) - mask; zp = torch.stack([F.softmax(z - mask, 1) for z in zs]).mean(0).clamp_min(1e-8).log()
    for tag, z in (("moy. logits", zl), ("moy. probas", zp)):
        for d in (0.0, 3.0, 6.0):
            zz = z.clone(); zz[torch.arange(n), yF] -= d; r = q(zz); OUT[f"{name}|{tag}|d{d}"] = r
            log(f"{name:34} {tag} delta {d:.0f} : {r[0]:.2f} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} +- {r[5]:.2f}")
best = sorted(OUT.items(), key=lambda kv: kv[1][1])[:12]
log("=== 12 meilleures cibles pour les classes partielles (KL partielles croissant)")
for k_, r in best: log(f"{k_:60} KL part {r[1]:.2f} | acc {r[2]:.2f} | accord gold {r[3]:.2f} | KL ent {r[0]:.2f}")
json.dump(OUT, open(V / "logs" / "replica5.json", "w"), indent=1); log("fini")
