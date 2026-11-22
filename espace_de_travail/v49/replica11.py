#!/usr/bin/env python
"""v49 - RETAIN : comment desapprendre l'image exacte SANS changer les reponses sur des images fraiches de la meme classe ?
Dans toutes nos maquettes le retain est le facteur le plus faible (0,5 a 0,6) : l'oubli appris sur les images a oublier
"deborde" sur les images de controle des memes classes. Nos ancrages (photos COCO, mosaiques) ne couvrent pas ce cas.
Idee : un vrai oubli du par-coeur ne devrait changer que l'image EXACTE (et ses copies legeres). On ajoute donc un ancrage sur
des vues FORTEMENT modifiees des images a oublier (la ou l'original repond deja "comme sur une image fraiche") : l'eleve doit
y rester identique a l'original. On mesure retain / prediction / mean_std avec un vrai gold.

(base : maquette v48)
v48 - Maquette "version 2" : classes LOURDES presque videes (pas retirees en entier), comme le suggere le leaderboard.
Questions : (1) qui porte la loss enorme chez le gold ? (2) peut-on deviner ces images (fragilite) ? (3) vaut-il mieux
sacrifier des classes entieres (biais) ou des images choisies une par une (etape 2 a marge) ? Chaque strategie est notee
avec la formule officielle et un VRAI gold (3 reentrainements). Base = maquette realiste v21/v39 (10 blocs figes).
KEEP = part des images d'une classe lourde qui restent dans l'entrainement du gold."""
import sys, json, time, copy, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); V = B / "v49"
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





log(f"=== maquette v2 (retain) : NB = {NB}, classes lourdes {FULL_REAL}, il en reste {KEEP:.0%} chez le gold")
COCO = torch.from_numpy(np.load(B / "ext" / "coco_val_224.npy")).to(dev)
n = len(Fo); G = GOLD["c"]; ar = torch.arange(n, device=dev); P = ~fF; fc = full_cls.tolist()
base = torch.stack(ZV).mean(0)
kl = lambda t, z: F.kl_div(F.log_softmax(z, 1), t, reduction="batchmean", log_target=True)
def coco_tok(k): return lower(U.augment(COCO[torch.randint(0, len(COCO), (k,), device=dev)].permute(0, 3, 1, 2).float() / 255, 1.0))
def cutmix(idx4):                                   # collage de 2 images a oublier (moitie gauche / moitie droite), taille reelle
    a_, b_ = unit(idx4[:, 0]), unit(idx4[:, 1]); x = a_.clone(); x[:, :, :, 112:] = b_[:, :, :, 112:]; return x
def student(delta=6.0, epochs=5, lr=5e-5, w_coco=1.0, view=0.0, w_view=0.0, view_T=2.0, w_mos=0.0, w_cut=0.0, aug_tr=0.25, hi_view=None, tgt_from="views", clip=None, ctrl_anchor=0.0):
    torch.manual_seed(0); src = base if tgt_from == "views" else vo["c"]; t = src.clone()
    if clip is None: t[ar, yF] -= delta
    else:
        o_ = t.clone(); o_[ar, yF] = -1e9; t[ar, yF] = o_.max(1).values + torch.minimum(t[ar, yF] - o_.max(1).values, torch.tensor(clip, device=dev))
    tl = F.log_softmax(t, 1); s = copy.deepcopy(orig); opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        s.train(); perm = torch.randperm(n)
        for b in range(0, n, 48):
            j = perm[b:b + 48]; jj = j.to(dev); k_ = len(j); toks = [TOK_F[jj], lower(U.augment(unit(Fo[j]), aug_tr))]; anchors = []
            if w_coco > 0: anchors.append((w_coco, coco_tok(k_)))
            if w_view > 0:
                a_ = view if hi_view is None else view + (hi_view - view) * torch.rand(1).item(); anchors.append((w_view, lower(U.augment(unit(Fo[j]), a_))))
            if w_mos > 0: anchors.append((w_mos, lower(mosaic(Fo[torch.randint(0, n, (k_, 4))]))))
            if w_cut > 0: anchors.append((w_cut, lower(cutmix(Fo[torch.randint(0, n, (k_, 2))]))))
            if ctrl_anchor > 0: anchors.append((ctrl_anchor, TOK_H[torch.randint(0, len(TOK_H), (k_,), device=dev)]))      # TRICHE (images de controle) : borne haute
            X = torch.cat(toks + [a for _, a in anchors])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): To = orig(X[2 * k_:]).float() if anchors else None
            with torch.autocast("cuda", dtype=torch.bfloat16): o = s(X).float()
            L = kl(tl[jj], o[:k_]) + kl(tl[jj], o[k_:2 * k_])
            for i_, (w_, _) in enumerate(anchors):
                sl = slice((2 + i_) * k_, (3 + i_) * k_); L = L + w_ * view_T * view_T * kl(F.log_softmax(To[i_ * k_:(i_ + 1) * k_] / view_T, 1), o[sl] / view_T)
            opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0); opt.step()
    return s.eval()
def shifted(v, spec):
    out = {}
    for k_, z in v.items():
        z = z.clone()
        for c, b in spec.items(): z[:, c] -= b
        out[k_] = z
    return out
po = F.log_softmax(vo["h"] / 2, 1); hF = is_full[H.to(dev)]
def rep3(name, s):
    v = views(s); r0 = score(v); best = None
    for tau in (1.0, 0.8):
        vt = {k_: z / tau for k_, z in v.items()}
        for b in range(0, 16, 2):
            vb = shifted(vt, {c: b for c in fc[:2]}); r = score(vb)
            if best is None or r["score_all"] > best[0]: best = (r["score_all"], b, tau, r)
    _, b, tau, r = best; ps = F.log_softmax(v["h"] / 2, 1); klh = (po.exp() * (po - ps)).sum(1); acc_s = (v["h"].argmax(1) == yH).float(); acc_o = (vo["h"].argmax(1) == yH).float()
    RES[name] = dict(r, beta=b, tau=tau, ret_nobias=r0["ret_all"], pred_nobias=(r0["pr_c"] + r0["pr_a"]) / 2)
    log(f"{name:50} sans biais : pred {r0['pr_c']:.2f}/{r0['pr_a']:.2f} RETAIN {r0['ret_all']:.3f} (KL controle lourdes {klh[hF].mean():.2f} partielles {klh[~hF].mean():.2f} ; acc controle {acc_s.mean():.3f} / orig {acc_o.mean():.3f}) "
        f"| meilleur (biais {b} sur 2 classes, logits/{tau}) : SCORE {r['score_all']:.3f} = ms {r['ms_c']:.2f}/{r['ms_a']:.2f} x pred {r['pr_c']:.2f}/{r['pr_a']:.2f} x retain {r['ret_all']:.2f} | loss part {r['L_part']:.2f}")
rep3("R0. reference : delta 6, ancrage COCO 1", student())
rep3("R0b. sans aucun ancrage", student(w_coco=0.0))
for a_, w_ in ((1.0, 1.0), (1.0, 3.0), (1.5, 1.0), (1.5, 3.0), (0.75, 1.0)):
    rep3(f"R1. + vues fortes {a_} ancrees a l'original, poids {w_}", student(view=a_, w_view=w_))
rep3("R1b. vues 1.0 a 1.5 (force tiree au hasard), poids 3", student(view=1.0, hi_view=1.5, w_view=3.0))
rep3("R1c. vues fortes 1.0 poids 3, ancrage a T=1", student(view=1.0, w_view=3.0, view_T=1.0))
rep3("R1d. vues fortes 1.0 poids 3, SANS COCO", student(view=1.0, w_view=3.0, w_coco=0.0))
rep3("R2. + mosaiques poids 1", student(w_mos=1.0)); rep3("R2. + mosaiques poids 3", student(w_mos=3.0))
rep3("R3. + collages de 2 images poids 3", student(w_cut=3.0))
rep3("R4. vues fortes 1.0 poids 3 + delta 8", student(view=1.0, w_view=3.0, delta=8.0))
rep3("R4. vues fortes 1.0 poids 3 + delta 10", student(view=1.0, w_view=3.0, delta=10.0))
rep3("R5. vues fortes 1.0 poids 3, 10 epoques", student(view=1.0, w_view=3.0, epochs=10))
rep3("R6. marge plafonnee a 3 (rappel v44)", student(clip=3.0))
rep3("R6. marge plafonnee a 3 + vues fortes 1.0 poids 3", student(clip=3.0, view=1.0, w_view=3.0))
rep3("R9. TRICHE : ancrage sur les images de controle (borne haute)", student(ctrl_anchor=3.0))
json.dump({k_: {a: (b if isinstance(b, (int, float)) else None) for a, b in r.items()} for k_, r in RES.items()}, open(V / "logs" / f"replica11_keep{KEEP}.json", "w"), indent=1); log("fini")
