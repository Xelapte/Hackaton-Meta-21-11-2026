#!/usr/bin/env python
"""v20 - MAQUETTE LOCALE de l'epreuve : on se fabrique un "original" et de VRAIS "gold" pour mesurer le vrai score.

Probleme : on ne voit jamais le gold, donc on regle tout a l'aveugle avec 6 soumissions par heure.
Idee : rejouer l'epreuve en petit, avec les 3500 images qu'on possede (45 classes, labels = predictions de l'original).
  1. On met de cote 20 % des images (jeu de controle, jamais utilise a l'entrainement).
  2. "Original maquette" = petit reseau (ResNet-18, images 112 px) entraine sur le reste jusqu'au par-coeur.
  3. On choisit un "forget set maquette" de meme forme que le vrai : 4 classes retirees EN ENTIER + la moitie des
     images d'une vingtaine d'autres classes.
  4. "Gold maquette" = 3 reseaux reentraines SANS ce forget set (3 graines), comme le vrai gold (moyenne de plusieurs).
  5. On applique nos recettes d'oubli a l'original maquette (en n'utilisant que le forget set, comme en vrai) et on
     calcule la formule OFFICIELLE du score, cette fois avec un vrai gold : note d'oubli (images exactes + copies
     modifiees) x note de retain (jeu de controle).
Ce que ca apporte : savoir ce que fait VRAIMENT un gold (confiance sur une classe jamais vue, mauvaise classe choisie),
quel est le plafond avec une cible parfaite, et quelle recette gagne - sans depenser de soumission.
Limite : petit reseau et petites images -> les tendances se transposent, pas les valeurs exactes.
"""
import sys, json, time, copy, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, timm

B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v20"
sys.path.insert(0, str(B / "v19"))
import unlearn as U  # noqa: E402  (augment)
dev = "cuda"; t0 = time.time()
def log(*a):
    print(f"[{(time.time() - t0) / 60:5.1f} min]", *a, flush=True)

# ---------------------------------------------------------------- donnees
g = torch.Generator().manual_seed(0)
raw = torch.from_numpy(np.load(B / "forget_images.npz")["images"])
y_real = torch.load(B / "v8" / "cache" / "_meta.pt")["y"]
X = torch.cat([F.interpolate(raw[i:i + 500].to(dev).permute(0, 3, 1, 2).float() / 255, size=112, mode="bilinear",
                             antialias=True) for i in range(0, len(raw), 500)])        # [N,3,112,112] dans [0,1]
classes = y_real.unique().tolist(); K = len(classes)
y = torch.tensor([classes.index(int(c)) for c in y_real], device=dev)
FULL_REAL = [2, 5, 32, 36]                                   # classes "retirees en entier" dans la maquette
full_cls = torch.tensor([classes.index(c) for c in FULL_REAL], device=dev)
others = [k for k in range(K) if k not in full_cls.tolist()]
part_cls = torch.tensor(others[::2], device=dev)             # une classe sur deux perd la moitie de ses images
H, T, Fo = [], [], []
for k in range(K):
    idx = (y == k).nonzero().flatten().cpu(); idx = idx[torch.randperm(len(idx), generator=g)]
    nh = max(1, round(0.2 * len(idx))); H += idx[:nh].tolist(); tr = idx[nh:].tolist(); T += tr
    if k in full_cls.tolist(): Fo += tr
    elif k in part_cls.tolist(): Fo += tr[: len(tr) // 2]
H, T, Fo = torch.tensor(H), torch.tensor(T), torch.tensor(Fo)
R = torch.tensor(sorted(set(T.tolist()) - set(Fo.tolist())))
is_full = torch.isin(y, full_cls)
log(f"{K} classes | controle {len(H)} | entrainement {len(T)} | forget {len(Fo)} (dont {int(is_full[Fo].sum())} de classes entieres) | retain {len(R)}")

def norm(x): return (x - 0.5) / 0.5

def train(idx, seed, epochs=150):
    torch.manual_seed(seed)
    m = timm.create_model("resnet18", num_classes=K).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.05)
    steps = epochs * math.ceil(len(idx) / 128); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        m.train(); perm = idx[torch.randperm(len(idx))]
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(m(norm(U.augment(X[i], 0.6))).float(), y[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return m.eval()

@torch.no_grad()
def logits(m, idx, aug=0.0, seed=None, shift=0.0):
    m.eval(); out = []
    if seed is not None: torch.manual_seed(seed)
    for b in range(0, len(idx), 500):
        x = X[idx[b:b + 500].to(dev)]
        if aug > 0: x = U.augment(x, aug)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(m(norm(x)).float())
    z = torch.cat(out)
    if shift: z = z.clone(); z[:, full_cls] -= shift
    return z

def views(m, shift=0.0):
    return {"c": logits(m, Fo, shift=shift), "a": logits(m, Fo, aug=0.5, seed=77, shift=shift), "h": logits(m, H, shift=shift)}

orig = train(T, 0); log("original maquette entraine")
golds = [train(R, s) for s in (1, 2, 3)]; log("3 gold maquette entraines")
yF, yH = y[Fo.to(dev)], y[H.to(dev)]
fF = is_full[Fo.to(dev)]; hNF = ~is_full[H.to(dev)]
vo = views(orig); vg = [views(m) for m in golds]
ce = lambda z, yy: F.cross_entropy(z, yy, reduction="none")
GOLD = {}
for k in ("c", "a"):
    GOLD[k] = {"p": torch.stack([F.softmax(v[k] / 2, 1) for v in vg]).mean(0),
               "mean": torch.stack([ce(v[k], yF).mean() for v in vg]).mean(), "std": torch.stack([ce(v[k], yF).std() for v in vg]).mean()}

def score(v):
    r = {}
    for k in ("c", "a"):
        G = GOLD[k]
        d = lambda z: (ce(z, yF).mean() - G["mean"]).abs() + (ce(z, yF).std() - G["std"]).abs()
        kl = lambda z: (G["p"] * (G["p"].clamp_min(1e-9).log() - F.log_softmax(z / 2, 1))).sum(1).mean()
        r["ms_" + k] = (1 - (d(v[k]) / d(vo[k])).clamp(0, 1)).item(); r["pr_" + k] = (1 - (kl(v[k]) / kl(vo[k])).clamp(0, 1)).item()
    r["forget"] = (r["ms_c"] + r["ms_a"]) / 2 * (r["pr_c"] + r["pr_a"]) / 2
    po = F.log_softmax(vo["h"] / 2, 1); ps = F.log_softmax(v["h"] / 2, 1)
    for name, msk in (("ret_all", torch.ones_like(hNF)), ("ret_nofull", hNF)):
        acc_s = (v["h"][msk].argmax(1) == yH[msk]).float().mean(); acc_o = (vo["h"][msk].argmax(1) == yH[msk]).float().mean()
        klr = (po[msk].exp() * (po[msk] - ps[msk])).sum(1).mean()
        r[name] = ((acc_s / acc_o).clamp(0, 1) / (1 + klr)).item()
    r["score_all"] = r["forget"] * r["ret_all"]; r["score_nofull"] = r["forget"] * r["ret_nofull"]
    r["acc_full"] = (v["c"][fF].argmax(1) == yF[fF]).float().mean().item(); r["acc_part"] = (v["c"][~fF].argmax(1) == yF[~fF]).float().mean().item()
    r["acc_ctrl"] = (v["h"].argmax(1) == yH).float().mean().item()
    return r

RES = {}
def rep(name, v):
    r = score(v); RES[name] = r
    log(f"{name:34} SCORE {r['score_all']:.3f} (sans cl. entieres dans le controle {r['score_nofull']:.3f}) | oubli {r['forget']:.2f} = "
        f"ms {r['ms_c']:.2f}/{r['ms_a']:.2f} x pred {r['pr_c']:.2f}/{r['pr_a']:.2f} | retain {r['ret_all']:.2f}/{r['ret_nofull']:.2f} | "
        f"acc entieres {r['acc_full']:.2f} partielles {r['acc_part']:.2f} controle {r['acc_ctrl']:.2f}")

# ---------------------------------------------------------------- 1) ce que fait un vrai gold
log("=== DIAGNOSTIC : original vs gold (maquette)")
log(f"original : acc forget {(vo['c'].argmax(1) == yF).float().mean():.3f} loss {ce(vo['c'], yF).mean():.3f} | acc controle {(vo['h'].argmax(1) == yH).float().mean():.3f}")
for j, v in enumerate(vg):
    Lc = ce(v["c"], yF); p = F.softmax(v["c"], 1)
    log(f"gold {j} : acc entieres {(v['c'][fF].argmax(1) == yF[fF]).float().mean():.3f} partielles {(v['c'][~fF].argmax(1) == yF[~fF]).float().mean():.3f} | "
        f"loss entieres {Lc[fF].mean():.2f}+-{Lc[fF].std():.2f} partielles {Lc[~fF].mean():.2f}+-{Lc[~fF].std():.2f} | "
        f"confiance (maxprob) entieres {p[fF].max(1).values.mean():.2f} partielles {p[~fF].max(1).values.mean():.2f} | acc controle {(v['h'].argmax(1) == yH).float().mean():.3f}")
log(f"gold moyenne : loss {GOLD['c']['mean']:.2f} +- {GOLD['c']['std']:.2f} | proba moyenne (T=2) de la vraie classe : entieres "
    f"{GOLD['c']['p'][fF].gather(1, yF[fF][:, None]).mean():.4f} partielles {GOLD['c']['p'][~fF].gather(1, yF[~fF][:, None]).mean():.3f}")
pg = GOLD["c"]["p"]
for c in full_cls.tolist():
    q = pg[yF == c].mean(0); v_, i_ = q.topk(3)
    qo = F.softmax(logits(orig, Fo[(yF == c).cpu()], aug=1.0, seed=5), 1); qo[:, full_cls] = 0; qo = (qo / qo.sum(1, keepdim=True)).mean(0); vo_, io_ = qo.topk(3)
    log(f"classe entiere {classes[c]} : le gold repond {[classes[i] for i in i_.tolist()]} ({[round(x, 2) for x in v_.tolist()]}) | "
        f"l'original (vues modifiees, classes entieres masquees) penche vers {[classes[i] for i in io_.tolist()]} ({[round(x, 2) for x in vo_.tolist()]})")
rep("original (reference = 0)", vo)
rep("un gold seul vs la moyenne", vg[0])

# ---------------------------------------------------------------- 2) nos recettes sur l'original maquette
def mosaic(idx4):
    im = F.interpolate(X[idx4.flatten().to(dev)], size=56, mode="bilinear", antialias=True).view(-1, 4, 3, 56, 56)
    return torch.cat([torch.cat([im[:, 0], im[:, 1]], -1), torch.cat([im[:, 2], im[:, 3]], -1)], -2)

def unlearn(mode="delta", k=16, d_full=20.0, d_part=6.0, lam=0.5, w_ret=3.0, mx=False, epochs=8, lr=1e-4, target=None):
    torch.manual_seed(0); n = len(Fo)
    if target is None:
        if k > 0: zs = [logits(orig, Fo, aug=1.0, seed=1000 + j) for j in range(k)]
        else: zs = [vo["c"]]
        base = torch.stack(zs).mean(0)
        tgt = base.clone(); tgt[torch.arange(n), yF] -= torch.where(fF, torch.tensor(d_full, device=dev), torch.tensor(d_part, device=dev))
        tl = F.log_softmax(tgt, 1)
        if mode == "ref":
            P = torch.stack([F.softmax(z - 20.0 * F.one_hot(full_cls, K).sum(0), 1) for z in zs]).mean(0)
            Q = torch.stack([P[yF == c].mean(0) if (yF == c).any() else P.mean(0) for c in range(K)])
            ref = lam * P + (1 - lam) * Q[yF]; tl[fF] = ref[fF].clamp_min(1e-8).log(); tl = F.log_softmax(tl, 1)
        ta = tl
    else:
        tl, ta = target
    s = copy.deepcopy(orig); opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.0)
    pool = Fo[(~fF).cpu()] if mx else Fo
    kl = lambda t, z: F.kl_div(F.log_softmax(z, 1), t, reduction="batchmean", log_target=True)
    for ep in range(epochs):
        s.train(); perm = torch.randperm(n)
        for b in range(0, n, 48):
            j = perm[b:b + 48]; x = X[Fo[j].to(dev)]; xa = U.augment(x, 0.5); xm = mosaic(pool[torch.randint(0, len(pool), (len(j), 4))])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tm = F.log_softmax(orig(norm(xm)).float() / 2, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = s(norm(torch.cat([x, xa, xm]))).float()
            jj = j.to(dev); L = kl(tl[jj], o[:len(j)]) + kl(ta[jj], o[len(j):2 * len(j)]) + w_ret * 4 * kl(tm, o[2 * len(j):] / 2)
            opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step()
    return s.eval()

log("=== RECETTES (beta = extinction globale des classes entieres)")
for b in (4, 8, 12, 16): rep(f"original + beta {b} (sans entrainement)", views(orig, b))
CFG = {"delta 0 vue (v5)": dict(k=0, d_part=8), "delta 16 vues (record)": dict(k=16, d_part=6), "delta 16 vues, d_part 3": dict(k=16, d_part=3),
       "delta 16 vues, d_part 10": dict(k=16, d_part=10), "ref lambda 0.5 (v19)": dict(mode="ref", lam=0.5), "ref lambda 0 (v19)": dict(mode="ref", lam=0.0),
       "ref 0.5 + mosaiques sans cl. entieres": dict(mode="ref", lam=0.5, mx=True), "delta 16 vues, ancrage 1": dict(k=16, w_ret=1.0),
       "delta 16 vues, ancrage 10": dict(k=16, w_ret=10.0)}
for name, c in CFG.items():
    s = unlearn(**c)
    for b in (0, 4, 8):
        rep(f"{name} | beta {b}", views(s, b))
# plafond : cible = le VRAI gold (probabilites moyennes, images exactes et copies), meme entrainement
tl = (GOLD["c"]["p"].clamp_min(1e-8).log() * 2); ta = (GOLD["a"]["p"].clamp_min(1e-8).log() * 2)   # logits tels que softmax(./2) = p_gold
s = unlearn(target=(F.log_softmax(tl, 1), F.log_softmax(ta, 1)))
rep("PLAFOND : cible = vrai gold | beta 0", views(s, 0))
s = unlearn(target=(F.log_softmax(tl, 1), F.log_softmax(ta, 1)), mx=True, w_ret=1.0)
rep("PLAFOND, ancrage 1, mosaiques sans cl. entieres", views(s, 0))
json.dump(RES, open(V / "logs" / "replica.json", "w"), indent=1)
log("fini")
