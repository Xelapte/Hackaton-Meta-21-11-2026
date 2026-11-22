#!/usr/bin/env python
"""v41 - AMUN (Adversarial Machine UNlearning, Ebrahimpour-Boroojeny et al., ICML 2025) teste dans la maquette realiste.

Idee du papier : pour chaque image a oublier, on cherche l'exemple ADVERSE le plus proche (la plus petite retouche
invisible qui fait changer d'avis le modele). Deux informations en sortent, image par image :
  - la DISTANCE a parcourir (eps) : petite = l'image tient "par un fil" (apprise par coeur), grande = image typique ;
  - la CLASSE vers laquelle le modele bascule : la frontiere la plus proche, donc l'erreur la plus naturelle.
AMUN reentraine ensuite le modele sur ces exemples adverses avec leur (fausse) etiquette : la confiance sur les images a
oublier baisse comme chez un modele reentraine, sans toucher au reste. Fonctionne sans retain set.
Ici on mesure, avec un vrai gold : (1) eps predit-il les images que le gold rate ? (2) la classe adverse est-elle la
mauvaise classe du gold ? (3) le score final bat-il notre recette ?

(base : maquette v39)
v39 - PROFESSEUR "DEMEMORISE" : pipeline en deux temps, teste dans la maquette (base = v21).

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

B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v41"
sys.path.insert(0, str(B / "v19")); sys.path.insert(0, str(B))
import unlearn as U  # noqa: E402  (augment ; fait chdir vers unlearning/)
from task_template import build_model, load_model  # noqa: E402
dev = "cuda"; t0 = time.time(); import os
NB = int(os.environ.get("NB", "10")); NV = 6
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
s0 = unlearn()
for b in (0, 7, 10): rep(f"record (delta 16 vues) | beta {b}", views(s0, b))
n = len(Fo); G = GOLD["c"]
base = torch.stack(ZV).mean(0); tgt = base.clone()
tgt[torch.arange(n), yF] -= torch.where(fF, torch.tensor(20.0, device=dev), torch.tensor(6.0, device=dev)); rec_lp = F.log_softmax(tgt, 1)
g_ok = torch.stack([(v["c"].argmax(1) == yF) for v in vg]).float().mean(0) >= 0.5; g_arg = GOLD["c"]["p"].argmax(1); P = ~fF
def diag(name, zc):
    kl = (G["p"] * (G["p"].clamp_min(1e-9).log() - F.log_softmax(zc / 2, 1))).sum(1); L = ce(zc, yF); ok = zc.argmax(1) == yF
    gw = P & ~g_ok; gr = P & g_ok
    log(f"{name:40} KL au gold : entieres {kl[fF].mean():.2f} | partielles {kl[P].mean():.2f} (gold juste {kl[gr].mean():.2f}, gold faux {kl[gw].mean():.2f}) | acc part {ok[P].float().mean():.2f} "
        f"| gold FAUX : faux aussi {(~ok[gw]).float().mean():.2f}, meme classe {(zc.argmax(1)[gw] == g_arg[gw]).float().mean():.2f} | gold JUSTE : juste {ok[gr].float().mean():.2f} | loss part {L[P].mean():.2f}+-{L[P].std():.2f}")
log(f"gold (majorite) : juste sur {g_ok[P].float().mean():.2f} des images partielles")
diag("original", vo["c"]); diag("un vrai gold", vg[1]["c"]); diag("cible du record", rec_lp)

def lower_g(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = real.patch_embed(norm(x)); t = real._pos_embed(t); t = real.patch_drop(t); t = real.norm_pre(t)
        for blk in real.blocks[:NB]: t = blk(t)
    return t
def fwd_g(m, x):
    t = lower_g(x)
    with torch.autocast("cuda", dtype=torch.bfloat16): return m(t).float()
def pgd_l2(m, x, yv, eps, steps=20):
    delta = torch.zeros_like(x, requires_grad=True); alpha = 2.5 * eps / steps
    for _ in range(steps):
        loss = F.cross_entropy(fwd_g(m, (x + delta).clamp(0, 1)), yv); g_, = torch.autograd.grad(loss, delta)
        delta = delta + alpha * g_ / g_.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        delta = (delta * (eps / delta.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)).clamp(max=1.0)).detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()
EPSL = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
def build_adv(m):
    m.eval(); XA = torch.zeros(n, 3, 224, 224, dtype=torch.float16, device=dev); YA = yF.clone(); EP = torch.full((n,), 99.0, device=dev)
    for b in range(0, n, 48):
        idx = torch.arange(b, min(b + 48, n), device=dev); x = unit(Fo[idx.cpu()]); yv = yF[idx]; todo = torch.ones(len(idx), dtype=torch.bool, device=dev)
        for e in EPSL:
            if not todo.any(): break
            xa = pgd_l2(m, x[todo], yv[todo], e)
            with torch.no_grad(): pa = lg(m, lower(xa)).argmax(1)
            okk = pa != yv[todo]; ii = idx[todo]
            XA[ii[okk]] = xa[okk].half(); YA[ii[okk]] = pa[okk]; EP[ii[okk]] = e
            if e == EPSL[-1]: XA[ii[~okk]] = xa[~okk].half()
            t2 = todo.clone(); t2[todo] = ~okk; todo = t2
    return XA, YA, EP
XA, YA, EP = build_adv(orig)
log(f"exemples adverses construits : eps median {EP.median().item():.2f} | repartition " + " ".join(f"{e}:{int((EP == e).sum())}" for e in EPSL) + f" | echecs {int((EP > 50).sum())}")
gw = P & ~g_ok; gr = P & g_ok
def auc(score, pos, neg):
    a = score[pos][:, None]; b_ = score[neg][None, :]; return ((a > b_).float().mean() + 0.5 * (a == b_).float().mean()).item()
with torch.no_grad(): za = lg(orig, lower(XA.float()))
conf_adv = F.softmax(za, 1).max(1).values
m16 = base.gather(1, yF[:, None]).squeeze(1) - base.scatter(1, yF[:, None], -1e9).max(1).values
log(f"eps moyen : images que le gold RATE {EP[gw].mean().item():.2f} / qu'il REUSSIT {EP[gr].mean().item():.2f} | AUC(petit eps -> gold rate) = {auc(-EP, gw, gr):.3f} "
    f"| AUC(marge sur 16 vues -> gold rate) = {auc(-m16, gw, gr):.3f}")
log(f"quand le gold rate : la classe adverse est SA mauvaise classe dans {(YA[gw] == g_arg[gw]).float().mean().item():.2f} des cas "
    f"(2e choix de l'original sur 16 vues : {(base.scatter(1, yF[:, None], -1e9).argmax(1)[gw] == g_arg[gw]).float().mean().item():.2f})")
for q in (0.1, 0.19, 0.3):
    thr = torch.quantile(EP[P], q); selq = P & (EP <= thr)
    log(f"  les {q:.0%} plus petits eps (<= {thr.item():.2f}) : le gold en rate {(~g_ok[selq]).float().mean().item():.2f} (contre {(~g_ok[P]).float().mean().item():.2f} en general)")
TOK_XA = torch.cat([lower(XA[i:i + 250].float()) for i in range(0, n, 250)])

def amun(epochs=5, lr=5e-5, w_ret=1.0, use_f=True, wa=1.0):
    torch.manual_seed(0); s = copy.deepcopy(orig); opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        s.train(); perm = torch.randperm(n)
        for b in range(0, n, 48):
            j = perm[b:b + 48]; jj = j.to(dev); toks = [TOK_XA[jj]]
            if use_f: toks.append(TOK_F[jj])
            if w_ret > 0:
                tm_tok = lower(mosaic(Fo[torch.randint(0, n, (len(j), 4))]))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): tm = F.log_softmax(orig(tm_tok).float() / 2, 1)
                toks.append(tm_tok)
            with torch.autocast("cuda", dtype=torch.bfloat16): o = s(torch.cat(toks)).float()
            L = wa * F.cross_entropy(o[:len(j)], YA[jj]); k_ = len(j)
            if use_f: L = L + F.cross_entropy(o[k_:2 * k_], yF[jj]); k_ = 2 * k_
            if w_ret > 0: L = L + w_ret * 4 * F.kl_div(F.log_softmax(o[k_:] / 2, 1), tm, reduction="batchmean", log_target=True)
            opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step()
    return s.eval()
log("=== AMUN : reentrainement sur les exemples adverses (+ images a oublier avec leur vraie etiquette)")
CFG = {"AMUN 3 epoques": dict(epochs=3), "AMUN 5 epoques": dict(), "AMUN 10 epoques": dict(epochs=10), "AMUN lr 2e-4": dict(lr=2e-4), "AMUN sans ancrage": dict(w_ret=0.0),
       "AMUN adverses seuls": dict(use_f=False), "AMUN adverses x3": dict(wa=3.0), "AMUN ancrage 3": dict(w_ret=3.0)}
for name, c in CFG.items():
    s = amun(**c); diag(f"  -> [{name}]", lg(s, TOK_F))
    for b in (0, 4, 7, 10): rep(f"{name} | beta {b}", views(s, b))

log("=== CIBLE INFORMEE : on fait basculer vers la classe adverse les images au plus petit eps, les autres restent justes")
def informed(q, pflip=0.75, pkeep=0.9):
    rp = rec_lp.exp().clone(); rp[torch.arange(n), yF] = 0; rp[:, full_cls] = 0; rp = rp / rp.sum(1, keepdim=True)          # repartition "plausible" du reste
    thr = torch.quantile(EP[P], q); fl = P & (EP <= thr); t = torch.zeros(n, K, device=dev)
    oh = lambda v: F.one_hot(v, K).float()
    t[P] = (pkeep * oh(yF) + (1 - pkeep) * rp)[P]
    t[fl] = (pflip * oh(YA) + 0.15 * oh(yF) + (1 - pflip - 0.15) * rp)[fl]
    tl = t.clamp_min(1e-6).log(); tl[fF] = rec_lp[fF]; tl = F.log_softmax(tl, 1); return tl, tl
for q in (0.1, 0.19, 0.3):
    t = informed(q); diag(f"cible informee q={q}", t[0])
    for kw_name, kw in (("", {}), (", ancrage 1, 10 ep", dict(w_ret=1.0, epochs=10))):
        s = unlearn(target=t, **kw); diag(f"  -> eleve q={q}{kw_name}", lg(s, TOK_F))
        for b in (0, 7, 10): rep(f"cible informee q={q}{kw_name} | beta {b}", views(s, b))
json.dump(RES, open(V / "logs" / f"replica8_nb{NB}.json", "w"), indent=1); log("fini")
