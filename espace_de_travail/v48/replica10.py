#!/usr/bin/env python
"""v48 - Maquette "version 2" : classes LOURDES presque videes (pas retirees en entier), comme le suggere le leaderboard.
Questions : (1) qui porte la loss enorme chez le gold ? (2) peut-on deviner ces images (fragilite) ? (3) vaut-il mieux
sacrifier des classes entieres (biais) ou des images choisies une par une (etape 2 a marge) ? Chaque strategie est notee
avec la formule officielle et un VRAI gold (3 reentrainements). Base = maquette realiste v21/v39 (10 blocs figes).
KEEP = part des images d'une classe lourde qui restent dans l'entrainement du gold."""
import sys, json, time, copy, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v48"
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
FULL_REAL = [0, 38, 6, 7, 30, 31]; KEEP = float(os.environ.get("KEEP", "0.15"))
full_cls = torch.tensor([classes.index(c) for c in FULL_REAL], device=dev)
others = [k for k in range(K) if k not in full_cls.tolist()]; part_cls = others[::2]
H, T, Fo = [], [], []
for k in range(K):
    idx = (y == k).nonzero().flatten().cpu(); idx = idx[torch.randperm(len(idx), generator=g)]
    nh = max(1, round(0.2 * len(idx))); H += idx[:nh].tolist(); tr = idx[nh:].tolist(); T += tr
    if k in full_cls.tolist(): Fo += tr[round(KEEP * len(tr)):]          # classe LOURDE : presque tout part, il en reste KEEP
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




log(f"=== maquette v2 : NB = {NB}, classes lourdes {FULL_REAL}, il en reste {KEEP:.0%} chez le gold")
COCO = torch.from_numpy(np.load(B / "ext" / "coco_val_224.npy")).to(dev)
n = len(Fo); G = GOLD["c"]; ar = torch.arange(n, device=dev); P = ~fF
base = torch.stack(ZV).mean(0); oth = base.clone(); oth[ar, yF] = -1e9; marg = base[ar, yF] - oth.max(1).values
Lg = torch.stack([ce(v["c"], yF) for v in vg]); gt2 = G["p"][ar, yF]; accg = torch.stack([(v["c"].argmax(1) == yF).float() for v in vg]).mean(0)
hi = (Lg > 8).float().mean(0)
log(f"GOLD lourdes : acc {accg[fF].mean():.2f} | loss {Lg[:, fF].mean():.2f}+-{Lg[:, fF].std():.2f} | part des loss > 8 : {(Lg[:, fF] > 8).float().mean():.2f} | proba T=2 de la vraie classe (moyenne des 3 gold) {gt2[fF].mean():.2f}")
log(f"GOLD partielles : acc {accg[P].mean():.2f} | loss {Lg[:, P].mean():.2f}+-{Lg[:, P].std():.2f} | part des loss > 8 : {(Lg[:, P] > 8).float().mean():.3f} | proba T=2 vraie {gt2[P].mean():.2f}")
log(f"GOLD ensemble : loss {GOLD['c']['mean']:.2f}+-{GOLD['c']['std']:.2f} | images a loss > 8 : {(Lg > 8).float().mean():.3f} du total, dont {((Lg > 8).float().mean(0) * fF).sum() / (Lg > 8).float().mean(0).sum():.2f} dans les classes lourdes")
def auc(sc, pos, neg):
    a = sc[pos][:, None]; b_ = sc[neg][None, :]; return ((a > b_).float().mean() + 0.5 * (a == b_).float().mean()).item()
bad = hi >= 0.5
log(f"fragilite (petite marge sur 16 vues) -> image a grosse loss chez le gold : AUC lourdes {auc(-marg, bad & fF, ~bad & fF):.3f} | toutes {auc(-marg, bad, ~bad):.3f} | correlation marge / proba T=2 du gold : {torch.corrcoef(torch.stack([marg, gt2]))[0, 1]:.2f}")
ZV15 = orig_views(8, 1.5); b15 = torch.stack(ZV15).mean(0); o15 = b15.clone(); o15[ar, yF] = -1e9; marg15 = b15[ar, yF] - o15.max(1).values
log(f"   avec des vues plus fortes (1.5) : AUC lourdes {auc(-marg15, bad & fF, ~bad & fF):.3f} | toutes {auc(-marg15, bad, ~bad):.3f}")

kl = lambda t, z: F.kl_div(F.log_softmax(z, 1), t, reduction="batchmean", log_target=True)
def coco_tok(k): return lower(U.augment(COCO[torch.randint(0, len(COCO), (k,), device=dev)].permute(0, 3, 1, 2).float() / 255, 1.0))
def base_student(delta=6.0, epochs=5, lr=5e-5, w_ret=1.0):
    torch.manual_seed(0); t = base.clone(); t[ar, yF] -= delta; tl = F.log_softmax(t, 1)
    s = copy.deepcopy(orig); opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        s.train(); perm = torch.randperm(n)
        for b in range(0, n, 48):
            j = perm[b:b + 48]; jj = j.to(dev); ta_tok = lower(U.augment(unit(Fo[j]), 0.25)); tm_tok = coco_tok(len(j))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): tm = F.log_softmax(orig(tm_tok).float() / 2, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16): o = s(torch.cat([TOK_F[jj], ta_tok, tm_tok])).float()
            L = kl(tl[jj], o[:len(j)]) + kl(tl[jj], o[len(j):2 * len(j)]) + w_ret * 4 * kl(tm, o[2 * len(j):] / 2)
            opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step()
    return s.eval()
def others_lp(z, yy):
    z = z.clone(); z[torch.arange(len(yy), device=dev), yy] = -1e9; return F.log_softmax(z, 1)
def stage2(teacher, sac, margin, steps=600, lr=1.5e-5, w_sac=0.3, w_keep=4.0, bs=32):
    torch.manual_seed(1); s = copy.deepcopy(teacher); opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1); i_s = sac.nonzero().flatten().cpu(); i_o = (~sac).nonzero().flatten().cpu(); arb = torch.arange(bs, device=dev)
    for st in range(steps):
        s.train(); js = i_s[torch.randint(0, len(i_s), (bs,))]; jo = i_o[torch.randint(0, len(i_o), (bs,))]
        X = torch.cat([TOK_F[js.to(dev)], lower(U.augment(unit(Fo[js]), 0.25 + 0.25 * torch.rand(1).item())), TOK_F[jo.to(dev)], lower(U.augment(unit(Fo[jo]), 0.25)), coco_tok(bs)])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): T_ = teacher(X).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): S_ = s(X).float()
        ys = yF[js.to(dev)]; L = 0.0
        for k in (0, 1):
            z = S_[k * bs:(k + 1) * bs]; zo = z.clone(); zo[arb, ys] = -1e9; v_ = z[arb, ys] - torch.logsumexp(zo, 1) + margin
            L = L + w_sac * F.smooth_l1_loss(v_, torch.zeros_like(v_)) + F.kl_div(others_lp(z, ys), others_lp(T_[k * bs:(k + 1) * bs], ys), reduction="batchmean", log_target=True)
        L = L + w_keep * (kl(F.log_softmax(T_[2 * bs:3 * bs], 1), S_[2 * bs:3 * bs]) + kl(F.log_softmax(T_[3 * bs:4 * bs], 1), S_[3 * bs:4 * bs])) + 4.0 * kl(F.log_softmax(T_[4 * bs:] / 2, 1), S_[4 * bs:] / 2)
        opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step(); sched.step()
    return s.eval()
def shifted(v, spec):
    out = {}
    for k_, z in v.items():
        z = z.clone()
        for c, b in spec.items(): z[:, c] -= b
        out[k_] = z
    return out
def rep2(name, v, sac=None):
    r = score(v); RES[name] = r; zc = v["c"]; klv = (G["p"] * (G["p"].clamp_min(1e-9).log() - F.log_softmax(zc / 2, 1))).sum(1); L = ce(zc, yF); extra = ""
    if sac is not None: extra = f" | sacrifiees : loss {L[sac].mean():.1f}, KL {klv[sac].mean():.2f}, proba gold T=2 {gt2[sac].mean():.2f} ; autres : loss {L[~sac].mean():.2f}, KL {klv[~sac].mean():.2f}"
    log(f"{name:44} SCORE {r['score_all']:.3f} | ms {r['ms_c']:.2f}/{r['ms_a']:.2f} pred {r['pr_c']:.2f}/{r['pr_a']:.2f} retain {r['ret_all']:.2f} | m/s {L.mean():.2f}/{L.std():.2f} | KL lourdes {r['kl_full']:.2f} part {r['kl_part']:.2f}" + extra)
    return r
klo = (G["p"] * (G["p"].clamp_min(1e-9).log() - F.log_softmax(vo["c"] / 2, 1))).sum(1)
log(f"reference : KL(gold || original) lourdes {klo[fF].mean():.2f} partielles {klo[P].mean():.2f} | poids des lourdes dans le denominateur : {klo[fF].sum() / klo.sum():.2f}")
S0 = base_student(); v0 = views(S0); rep2("A. eleve delta 6, aucun sacrifice", v0)
# original + biais seul (equivalent de orig_a28)
fc = full_cls.tolist()
log("=== B. sacrifice par CLASSES entieres (biais), beta choisi au mieux (oracle)")
def best_bias(vbase, cls, name, betas=range(4, 34, 2)):
    best = None
    for b in betas:
        v = shifted(vbase, {c: b for c in cls}); r = score(v)
        if best is None or r["score_all"] > best[0]: best = (r["score_all"], b, v)
    sacm = torch.isin(yF, torch.tensor(cls, device=dev)); rep2(f"{name} (beta {best[1]})", best[2], sacm); return best
best_bias(vo, fc[:2], "B0. ORIGINAL + biais sur 2 classes lourdes"); best_bias(vo, fc, "B0. ORIGINAL + biais sur les 6 lourdes")
for cls, nm in ((fc[:2], "B1. eleve + biais classes 0/38"), (fc[2:4], "B2. eleve + biais 2 lanternes"), (fc[4:6], "B3. eleve + biais 2 bateaux"), (fc, "B4. eleve + biais 6 lourdes"), (fc[:1], "B5. eleve + biais 1 classe")):
    best_bias(v0, cls, nm)
log("=== C. sacrifice IMAGE PAR IMAGE (etape 2 a marge), meme nombre d'images et meme loss visee pour tous")
L0 = ce(v0["c"], yF); mu_g, sg_g = GOLD["c"]["mean"].item(), GOLD["c"]["std"].item(); bestq = None
for q in [x / 100 for x in range(2, 31)]:
    for M in range(6, 26):
        k_ = int(q * n); m_ = (L0.sum().item() - L0.topk(k_).values.sum().item() * 0 + k_ * (M - L0.mean().item())) / n
        e2 = ((L0 ** 2).sum().item() + k_ * (M * M - (L0 ** 2).mean().item())) / n; s_ = max(e2 - m_ * m_, 0) ** 0.5; d_ = abs(m_ - mu_g) + abs(s_ - sg_g)
        if bestq is None or d_ < bestq[0]: bestq = (d_, q, M)
_, q, M = bestq; k_sac = int(q * n); log(f"pour imiter la loss du gold ({mu_g:.2f}+-{sg_g:.2f}) : sacrifier {q:.0%} des images ({k_sac}) a loss {M}")
def topk_mask(sc, pool): sc = sc.clone(); sc[~pool] = 1e9; m = torch.zeros(n, dtype=torch.bool, device=dev); m[sc.topk(k_sac, largest=False).indices] = True; return m
allm = torch.ones(n, dtype=torch.bool, device=dev); torch.manual_seed(5)
SETS = {"C1. plus fragiles parmi les lourdes": topk_mask(marg, fF if int(fF.sum()) >= k_sac else allm), "C2. plus fragiles, toutes classes": topk_mask(marg, allm),
        "C3. ORACLE : celles que le gold rate le plus": topk_mask(gt2, allm), "C4. au hasard parmi les lourdes": topk_mask(torch.rand(n, device=dev), fF if int(fF.sum()) >= k_sac else allm),
        "C5. fragiles (vues 1.5), toutes classes": topk_mask(marg15, allm)}
for nm, sac in SETS.items():
    log(f"   {nm} : {int((sac & fF).sum())} lourdes sur {k_sac} ; proba T=2 du gold sur ces images {gt2[sac].mean():.2f} (moyenne generale {gt2.mean():.2f}) ; part a grosse loss chez le gold {hi[sac].mean():.2f}")
    s2 = stage2(S0, sac, float(M)); rep2(nm, views(s2), sac)
    if nm.startswith("C3"):
        for tau in (0.8, 0.65): rep2(f"   C3 + logits / {tau}", {k_: z / tau for k_, z in views(s2).items()}, sac)
log("=== D. temperature sur le meilleur reglage par classes")
bb = best_bias(v0, fc[:2], "D0. rappel B1")
for tau in (0.8, 0.65, 1.25):
    vt = {k_: z / tau for k_, z in v0.items()}; best_bias(vt, fc[:2], f"D. logits / {tau} + biais 0/38")
json.dump({k_: {a: (b if isinstance(b, (int, float)) else None) for a, b in r.items()} for k_, r in RES.items()}, open(V / "logs" / f"replica10_keep{KEEP}.json", "w"), indent=1); log("fini")
