#!/usr/bin/env python
"""v8 - Simulateur du score, calibre sur nos vraies notes du leaderboard. CPU, quelques secondes.

Principe :
  1. "Faux gold" ajustable, fabrique a partir des reponses de l'original sur l'image exacte :
       - on retire dp au logit de la vraie classe (on enleve le par-coeur),
       - on retire a aux logits des 9 classes entieres, sur toutes les images (classes jamais vues),
       - on divise par tau (le gold est une MOYENNE de plusieurs modeles -> plus hesitant).
  2. On rejoue la formule officielle (softmax T=2) : forget = mean_std x prediction, sur images exactes
     et sur copies modifiees ; retain ~ 1 / (1 + g * KL(original || nous)) mesure sur des mosaiques.
  3. On cherche les reglages (a, dp, tau, g, force des copies) qui reproduisent le mieux nos notes reelles (lb.json).
  4. Avec les meilleurs reglages, on note tous les candidats (modele x beta) et on affiche le classement.
C'est une BOUSSOLE (peu de points de calibration), pas une verite : elle sert a trier, le leaderboard decide.
Usage : python sim.py [--top 20]
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
VDIR = Path(__file__).resolve().parent
C = VDIR / "cache"


def lsm2(z):
    return F.log_softmax(z.float() / 2, -1)


def forget_parts(ls_sub, ls_gold, ls_orig, y):
    """mean_std et prediction (formule officielle) a partir de log-probas T=2."""
    def L(ls):
        return -ls.gather(1, y[:, None]).squeeze(1)

    def d(ls):
        return (L(ls).mean() - L(ls_gold).mean()).abs() + (L(ls).std() - L(ls_gold).std()).abs()

    def kl(ls):
        return (ls_gold.exp() * (ls_gold - ls)).sum(1).mean()

    ms = 1 - (d(ls_sub) / d(ls_orig).clamp_min(1e-8)).clamp(0, 1)
    pr = 1 - (kl(ls_sub) / kl(ls_orig).clamp_min(1e-8)).clamp(0, 1)
    return ms.item(), pr.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--nfit", type=int, default=10, help="nb de meilleurs calibrages moyennes pour noter les candidats")
    args = ap.parse_args()

    meta = torch.load(C / "_meta.pt")
    y, full, FULL = meta["y"], meta["full"], meta["full_cls"]
    N = len(y)
    orig = torch.load(C / "orig.pt")
    names = sorted(p.stem for p in C.glob("*.pt") if p.stem not in ("_meta", "orig"))
    cache = {n: torch.load(C / f"{n}.pt") for n in names}
    cache["orig"] = orig
    lb = {k: v for k, v in json.load(open(VDIR / "lb.json")).items() if not k.startswith("_")}

    y2 = y.repeat(2)
    lo = {"clean": lsm2(orig["clean"]), "aug05": lsm2(orig["aug05"].flatten(0, 1)), "aug10": lsm2(orig["aug10"].flatten(0, 1)),
          "mos": lsm2(orig["mos_part"])}

    def sub_views(name, beta):
        c = cache[name]
        def sh(z):
            z = z.float().clone(); z[..., FULL] -= beta; return z
        return {"clean": lsm2(sh(c["clean"])), "aug05": lsm2(sh(c["aug05"]).flatten(0, 1)),
                "aug10": lsm2(sh(c["aug10"]).flatten(0, 1)), "mos": lsm2(sh(c["mos_part"]))}

    def gold(a, dp, tau):
        g = orig["clean"].float().clone()
        g[torch.arange(N), y] -= dp
        g[:, FULL] -= a
        ls = lsm2(g / tau)
        return ls, ls.repeat(2, 1)

    def sim(sv, g, alt, gam):
        g1, g2 = g
        ms_c, pr_c = forget_parts(sv["clean"], g1, lo["clean"], y)
        ms_a, pr_a = forget_parts(sv[alt], g2, lo[alt], y2)
        kl_ret = (lo["mos"].exp() * (lo["mos"] - sv["mos"])).sum(1).mean().item()
        return (ms_c + ms_a) / 2 * (pr_c + pr_a) / 2 / (1 + gam * kl_ret)

    # ---- 1) calibration sur les notes reelles
    pts = [(k.split("@")[0], float(k.split("@")[1]), v) for k, v in lb.items() if k.split("@")[0] in cache]
    print(f"{len(pts)} notes de calibration : " + ", ".join(f"{n}@{b:g}={v}" for n, b, v in pts))
    svs = [sub_views(n, b) for n, b, _ in pts]
    grid = itertools.product([2, 4, 6, 8, 10, 12, 16], [4, 6, 8, 10, 12, 14], [1.0, 1.5, 2.0, 3.0], ["aug05", "aug10"], [0.0, 0.3, 1.0])
    golds, fits = {}, []
    for a, dp, tau, alt, gam in grid:
        if (a, dp, tau) not in golds:
            golds[(a, dp, tau)] = gold(a, dp, tau)
        s = [max(sim(sv, golds[(a, dp, tau)], alt, gam), 1e-4) for sv in svs]
        dl = [math.log(v) - math.log(si) for (_, _, v), si in zip(pts, s)]
        c = sum(dl) / len(dl)                                   # facteur d'echelle global (log)
        err = math.sqrt(sum((x - c) ** 2 for x in dl) / len(dl))
        fits.append((err, a, dp, tau, alt, gam, math.exp(c), s))
    fits.sort(key=lambda t: t[0])
    print("\nMeilleurs calibrages (err = ecart moyen en log ; 0.2 ~ 20 % d'erreur) :")
    print(f"{'err':>5} {'a':>3} {'dp':>3} {'tau':>4} {'copies':>6} {'g':>4} {'ech.':>5} | simule (x echelle) vs reel")
    for err, a, dp, tau, alt, gam, c, s in fits[:8]:
        print(f"{err:5.2f} {a:3d} {dp:3d} {tau:4.1f} {alt:>6} {gam:4.1f} {c:5.2f} | " +
              " ".join(f"{si * c:.3f}/{v:.3f}" for si, (_, _, v) in zip(s, pts)))

    # ---- 2) classement des candidats avec la moyenne des nfit meilleurs calibrages
    best = fits[:args.nfit]
    rows = []
    for n in names + ["orig"]:
        for beta in [0, 2, 4, 5, 6, 7, 8, 9, 10, 12]:
            sv = sub_views(n, beta)
            sc = [sim(sv, golds[(a, dp, tau)], alt, gam) * c for _, a, dp, tau, alt, gam, c, _ in best]
            kl_ret = (lo["mos"].exp() * (lo["mos"] - sv["mos"])).sum(1).mean().item()
            agree = (sv["mos"].argmax(1) == lo["mos"].argmax(1)).float().mean().item()
            rows.append((sum(sc) / len(sc), min(sc), max(sc), n, beta, kl_ret, agree))
    rows.sort(reverse=True)
    print(f"\nClassement simule (moyenne des {len(best)} meilleurs calibrages) :")
    print(f"{'score':>6} {'min':>6} {'max':>6}  {'modele':12} {'beta':>4} | {'KL mos':>6} {'accord':>6} | reel")
    for m, mn, mx, n, beta, kl, ag in rows[:args.top]:
        print(f"{m:6.3f} {mn:6.3f} {mx:6.3f}  {n:12} {beta:4d} | {kl:6.3f} {ag:6.3f} | {lb.get(f'{n}@{beta}', '')}")
    json.dump([{"score": m, "min": mn, "max": mx, "model": n, "beta": b, "kl_mos": kl, "agree": ag} for m, mn, mx, n, b, kl, ag in rows],
              open(VDIR / "logs" / "sim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
