#!/usr/bin/env python
"""v39 - PROFESSEUR "DEMEMORISE" : pipeline en deux temps, teste dans la maquette (base = v21).

Constat : notre cible actuelle (reponses de l'original, vraie classe baissee) ne dit pas QUELLES images un gold rate ni
vers quelle classe. Idee :
  1. PROFESSEUR : copie de l'original dont les couches qui memorisent sont remises a zero, puis qui reapprend a imiter
     l'original UNIQUEMENT sur des images exterieures (photos COCO). Il n'a donc plus le par-coeur : sur une image a
     oublier il repond "par generalisation", avec ses propres erreurs, image par image.
     Seul, ce professeur est un mauvais candidat (il imite mal l'original ailleurs -> retain faible).
  2. ELEVE : on repart de l'ORIGINAL (retain intact) et on lui apprend les reponses du professeur sur les images a
     oublier seulement. Puis extinction beta des classes retirees en entier.
La maquette (vrai gold disponible) dit si les reponses du professeur sont plus proches du gold que notre cible, et si
le score final monte. NB = nombre de blocs du vrai modele gardes figes (variable d'environnement).

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

B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v39"
sys.path.insert(0, str(B / "v19")); sys.path.insert(0, str(B))
import unlearn as U  # noqa: E402  (augment ; fait chdir vers unlearning/)
from task_template import build_model, load_model  # noqa: E402
dev = "cuda"; t0 = time.time(); import os
NB = int(os.environ.get("NB", "8")); NV = 6
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
ZV = orig_views(16); log("16 vues modifiees de l'original calculees")

def unlearn(mode="delta", k=16, d_full=20.0, d_part=6.0, lam=0.5, w_ret=3.0, mx=False, epochs=5, lr=5e-5, target=None, tau=1.0, w_aug=1.0):
    torch.manual_seed(0); n = len(Fo)
    if target is None:
        zs = ZV[:k] if k > 0 else [vo["c"]]
        base = torch.stack(zs).mean(0); tgt = base.clone()
        tgt[torch.arange(n), yF] -= torch.where(fF, torch.tensor(d_full, device=dev), torch.tensor(d_part, device=dev))
        tl = F.log_softmax(tgt / tau, 1)
        if mode == "ref":
            P = torch.stack([F.softmax(z - 20.0 * F.one_hot(full_cls, K).sum(0), 1) for z in zs]).mean(0)
            Q = torch.stack([P[yF == c].mean(0) if (yF == c).any() else P.mean(0) for c in range(K)])
            ref = lam * P + (1 - lam) * Q[yF]; tl[fF] = ref[fF].clamp_min(1e-8).log(); tl = F.log_softmax(tl, 1)
        ta = tl
    else: tl, ta = target
    s = copy.deepcopy(orig); opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.0)
    pool = Fo[(~fF).cpu()] if mx else Fo
    kl = lambda t, z: F.kl_div(F.log_softmax(z, 1), t, reduction="batchmean", log_target=True)
    for ep in range(epochs):
        s.train(); perm = torch.randperm(n)
        for b in range(0, n, 48):
            j = perm[b:b + 48]; jj = j.to(dev); ta_tok = lower(U.augment(unit(Fo[j]), 0.5)); tm_tok = lower(mosaic(pool[torch.randint(0, len(pool), (len(j), 4))]))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): tm = F.log_softmax(orig(tm_tok).float() / 2, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16): o = s(torch.cat([TOK_F[jj], ta_tok, tm_tok])).float()
            L = kl(tl[jj], o[:len(j)]) + w_aug * kl(ta[jj], o[len(j):2 * len(j)]) + w_ret * 4 * kl(tm, o[2 * len(j):] / 2)
            opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step()
    return s.eval()


log(f"=== NB = {NB} blocs figes, {len(orig.blocks)} blocs entraines dans la maquette")
log("=== reference : recette du record")
s0 = unlearn()
for b in (0, 4, 7, 10): rep(f"record (delta 16 vues) | beta {b}", views(s0, b))
n = len(Fo); G = GOLD["c"]
base = torch.stack(ZV).mean(0); tgt = base.clone()
tgt[torch.arange(n), yF] -= torch.where(fF, torch.tensor(20.0, device=dev), torch.tensor(6.0, device=dev)); rec_lp = F.log_softmax(tgt, 1)
g_ok = torch.stack([(v["c"].argmax(1) == yF) for v in vg]).float().mean(0) >= 0.5            # le gold (majorite) a-t-il juste ?
g_arg = GOLD["c"]["p"].argmax(1)
def diag(name, zc):
    kl = (G["p"] * (G["p"].clamp_min(1e-9).log() - F.log_softmax(zc / 2, 1))).sum(1); L = ce(zc, yF); ok = zc.argmax(1) == yF; p = ~fF
    gw = p & ~g_ok; gr = p & g_ok
    log(f"{name:40} ecart au gold (KL) : entieres {kl[fF].mean():.2f} | partielles {kl[p].mean():.2f} (gold juste {kl[gr].mean():.2f}, gold faux {kl[gw].mean():.2f}) | acc part {ok[p].float().mean():.2f} "
        f"| quand le gold a FAUX : on a faux aussi {(~ok[gw]).float().mean():.2f}, meme classe que lui {(zc.argmax(1)[gw] == g_arg[gw]).float().mean():.2f} "
        f"| quand le gold a JUSTE : on a juste {ok[gr].float().mean():.2f} | loss part {L[p].mean():.2f}+-{L[p].std():.2f}")
log(f"gold (majorite des 3) : juste sur {g_ok[~fF].float().mean():.2f} des images des classes partielles")
log("--- qualite des CIBLES (plus l'ecart au gold est petit, mieux c'est)")
diag("original", vo["c"]); diag("un vrai gold (ideal)", vg[1]["c"]); diag("cible du record (delta 16 vues)", rec_lp)

COCO = torch.from_numpy(np.load(B / "ext" / "coco_val_224.npy")).to(dev)
def unit_c(idx): return COCO[idx].permute(0, 3, 1, 2).float() / 255
def make_teacher(seed, steps=1500, lr=5e-4, T=2.0):
    s = copy.deepcopy(orig); fresh = Top(seed + 50).to(dev)
    with torch.no_grad():
        s.blocks.load_state_dict(fresh.blocks.state_dict()); s.head.load_state_dict(fresh.head.state_dict()); s.norm.load_state_dict(fresh.norm.state_dict())
    opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    torch.manual_seed(seed); s.train()
    for st in range(steps):
        tok = lower(U.augment(unit_c(torch.randint(0, len(COCO), (96,), device=dev)), 1.0))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): zt = orig(tok).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): o = s(tok).float()
        L = F.kl_div(F.log_softmax(o / T, 1), F.log_softmax(zt / T, 1), reduction="batchmean", log_target=True) * T * T + F.kl_div(F.log_softmax(o, 1), F.log_softmax(zt, 1), reduction="batchmean", log_target=True)
        opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step(); sched.step()
    return s.eval()
log("=== PROFESSEURS dememorises (couches entrainees remises a zero, reapprentissage sur photos COCO)")
teachers = []
for sd_ in (1, 2, 3):
    t = make_teacher(sd_); teachers.append(t); diag(f"professeur {sd_}", lg(t, TOK_F))
    if sd_ == 1:
        for b in (0, 7): rep(f"professeur 1 soumis tel quel | beta {b}", views(t, b))
def tt(ts):
    pc = torch.stack([F.softmax(lg(t, TOK_F), 1) for t in ts]).mean(0).clamp_min(1e-8).log(); pa = torch.stack([F.softmax(lg(t, TOK_ALT), 1) for t in ts]).mean(0).clamp_min(1e-8).log()
    return F.log_softmax(pc, 1), F.log_softmax(pa, 1)
t1 = tt(teachers[:1]); t3 = tt(teachers)
diag("moyenne des 3 professeurs", t3[0])
long_t = make_teacher(7, steps=4000); diag("professeur long (4000 pas)", lg(long_t, TOK_F)); tL = tt([long_t])

log("=== ELEVE : original + reponses du professeur sur les images a oublier (+ extinction beta)")
def hyb(t):                                  # professeur pour les classes partielles, cible du record pour les classes entieres
    tl, ta = t[0].clone(), t[1].clone(); tl[fF] = rec_lp[fF]; ta[fF] = rec_lp[fF]; return tl, ta
RUNS = {"prof. 1": (t1, {}), "moyenne de 3 prof.": (t3, {}), "moyenne de 3, ancrage 1": (t3, dict(w_ret=1.0)), "moyenne de 3, 10 epoques": (t3, dict(epochs=10)),
        "moyenne de 3, lr 2e-4": (t3, dict(lr=2e-4)), "prof. long": (tL, {}),
        "hybride (prof. sur partielles, record sur entieres)": (hyb(t3), {}), "hybride, ancrage 1, 10 epoques": (hyb(t3), dict(w_ret=1.0, epochs=10))}
for name, (t, kw) in RUNS.items():
    s = unlearn(target=t, **kw); diag(f"  -> eleve [{name}]", lg(s, TOK_F))
    for b in (0, 4, 7, 10): rep(f"{name} | beta {b}", views(s, b))
json.dump(RES, open(V / "logs" / f"replica7_nb{NB}.json", "w"), indent=1); log("fini")
