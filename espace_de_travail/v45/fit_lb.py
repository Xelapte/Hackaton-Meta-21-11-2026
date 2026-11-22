#!/usr/bin/env python
"""v45 - Petit modele du leaderboard, ajuste sur nos propres notes (CPU).
Hypothese : note = c x mean_std(config) x retain(config), avec
  mean_std = moyenne sur (images exactes, copies) de 1 - (|m - MU| + |s - SIGMA|) / (MU + SIGMA)   [MU, SIGMA = gold, inconnus]
  retain   = (1 - part des images de controle des classes eteintes) / (1 + K0 + cout KL des classes eteintes)
Les losses de NOTRE modele (m, s) sont calculees exactement pour chaque reglage d'extinction a partir des sorties en cache."""
import sys, itertools, math, torch, torch.nn.functional as F
torch.set_num_threads(4)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"
c = torch.load(V + "cache.pt"); r = torch.load(V + "cache_record.pt"); y = c["y"]; n = len(y)
Ze = r["record ext_wr1"].float(); Za = r["record ext_wr1 | copies"].float(); Zo = r["original | copies"].float()
A, Bq, C = [0, 38], [6, 7, 8], [29, 30, 31, 33]
def stats(spec):
    out = []
    for Z in (Ze, Za):
        z = Z.clone()
        for cl, b in spec.items(): z[:, cl] -= b
        L = F.cross_entropy(z, y, reduction="none"); out += [L.mean().item(), L.std().item()]
    return out
def kcost(spec):                     # cout "retain" si le jeu de controle contient ces classes : KL(orig || eteint) a T=2 sur des copies, + perte de bonnes reponses
    k = 0.0; lost = 0.0
    for cl, b in spec.items():
        s = y == cl; zo = Zo[s]; zs = zo.clone(); zs[:, cl] -= b
        po = F.softmax(zo / 2, 1); k += 0.01 * (po * (F.log_softmax(zo / 2, 1) - F.log_softmax(zs / 2, 1))).sum(1).mean().item()
        lost += 0.01 * ((zo.argmax(1) == cl) & (zs.argmax(1) != cl)).float().mean().item()
    return k, lost
d = lambda keys, b: {k: b for k in keys}
DATA = [("A7 B7", {**d(A, 7), **d(Bq, 7)}, 0.1680), ("A10 B7", {**d(A, 10), **d(Bq, 7)}, 0.1865), ("A7 B4", {**d(A, 7), **d(Bq, 4)}, 0.1555), ("A10 B10", {**d(A, 10), **d(Bq, 10)}, 0.1894),
        ("tout 7", d(A + Bq + C, 7), 0.1552 * 1.012), ("sans B", d(A + C, 7), 0.1609 * 1.012), ("sans BC", d(A, 7), 0.1383 * 1.012), ("m0", d([38] + Bq, 7), 0.1499 * 1.012),
        ("m38", d([0] + Bq, 7), 0.1515 * 1.012), ("tout 5", d(A + Bq + C, 5), 0.1446 * 1.012), ("p29", d(A + Bq + [29], 7), 0.1648 * 1.012), ("cinq_a", d(C + [38], 7), 0.1460 * 1.012)]
EXTRA = [x for x in sys.argv[1:]]
for e in EXTRA:
    nm, a_, b_, sc = e.split(":"); DATA.append((nm, {**d(A, float(a_)), **d(Bq, float(b_))}, float(sc)))
ST = [stats(sp) for _, sp, _ in DATA]; KC = [kcost(sp) for _, sp, _ in DATA]
print("reglage      exactes (m, s)   copies (m, s)   cout KL  perte acc  note")
for (nm, sp, sc), st, kc in zip(DATA, ST, KC): print(f"{nm:10} {st[0]:5.2f} {st[1]:5.2f}   {st[2]:5.2f} {st[3]:5.2f}   {kc[0]:.3f}   {kc[1]:.3f}   {sc:.4f}")
def pred(par, st, kc, ctrl):
    MU, SG, K0, cst, DM, DS = par
    ms = 0.0
    for (m, s, mu, sg) in ((st[0], st[1], MU, SG), (st[2], st[3], MU + DM, SG + DS)):
        ms += 0.5 * (1 - min(1.0, (abs(m - mu) + abs(s - sg)) / (mu + sg)))
    ret = (1 - ctrl * kc[1] / 0.9) / (1 + K0 + ctrl * kc[0])
    return cst * ms * ret
CAND = {}
for a_, b_, c_ in itertools.product([7, 10, 13, 16, 20], [4, 7, 10, 13, 16], [0, 2, 4, 7, 10]):
    sp = {**d(A, a_), **d(Bq, b_), **(d(C, c_) if c_ else {})}; CAND[(a_, b_, c_)] = (stats(sp), kcost(sp))
print("candidats prets", flush=True)
best = {}
for ctrl in (0.0, 1.0):
    res = []
    for MU, SG, K0, DM, DS in itertools.product([x * 0.25 for x in range(4, 33)], [x * 0.25 for x in range(6, 45)], [0.0, 0.2, 0.6], [-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]):
        p0 = [pred((MU, SG, K0, 1.0, DM, DS), st, kc, ctrl) for st, kc in zip(ST, KC)]
        num = sum(p * sc for p, (_, _, sc) in zip(p0, DATA)); den = sum(p * p for p in p0)
        if den <= 0: continue
        cst = num / den; err = math.sqrt(sum((cst * p - sc) ** 2 for p, (_, _, sc) in zip(p0, DATA)) / len(DATA))
        res.append((err, MU, SG, K0, cst, DM, DS))
    res.sort(); best[ctrl] = res[:8]
    print(f"\n=== jeu de controle {'AVEC' if ctrl else 'SANS'} les classes eteintes : meilleurs ajustements (erreur, MU, SIGMA, K0, c, dM, dS)")
    for r_ in res[:8]: print("  err %.4f | MU %.2f SIGMA %.2f | K0 %.2f | c %.3f | copies : dM %+.1f dS %+.1f" % r_)
    e, MU, SG, K0, cst, DM, DS = res[0]
    print("  detail du meilleur :", " ".join(f"{nm}:{pred((MU, SG, K0, cst, DM, DS), st, kc, ctrl):.3f}/{sc:.3f}" for (nm, _, sc), st, kc in zip(DATA, ST, KC)))
    # predictions pour des reglages candidats
    cand = {}
    for key, (st_, kc_) in CAND.items():
        ps = [pred(r_[1:4] + (r_[4],) + r_[5:], st_, kc_, ctrl) for r_ in res[:8]]
        cand[key] = (sum(ps) / len(ps), min(ps), max(ps))
    top = sorted(cand.items(), key=lambda kv: -kv[1][0])[:10]
    print("  meilleurs reglages predits (A, B, C) : " + " | ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}-{v[2]:.3f}]" for k, v in top))
