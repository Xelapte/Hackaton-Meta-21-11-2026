"""Tâche 1 : analyse d'erreurs de H3b (CPU), avec des oracles qui bornent le gain de chaque correction.

Pour chaque image du train : prédiction H3b hors pli (seuils réglés sur les 4 autres plis), régime prédit
(full / partiel boîte / partiel vide) contre régime vrai (full_watermark), et plusieurs oracles :
  - régime parfait            : on applique la règle avec le vrai régime ;
  - meilleure des 20 boîtes   : parmi les 20 boîtes enregistrées, celle de meilleure IoU (borne d'un re-classement) ;
  - boîte vraie               : boîte englobante du masque (borne de la recherche de boîte) ;
  - masque vrai               : 1 (borne absolue).
Les oracles ne sont PAS des méthodes : ils disent où sont les points.

  python watermark/analyse_erreurs.py [--repli unet]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.protocole import K, METHODES, miou_lot  # noqa: E402
from watermark.regle import charger, masque_boite, predire, regler  # noqa: E402


def iou1(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--cartes", default=str(off / "output" / "cartes_feats.npz"))
    ap.add_argument("--repli", choices=["carte", "unet"], default="carte")
    ap.add_argument("--oof", default=str(off / "output" / "unet_h5a_feats_oof.npz"))
    ap.add_argument("--menus", action="store_true")
    args = ap.parse_args()
    D, meta = charger(args, "train")
    full_vrai = np.array([bool(m["full_watermark"]) for m in meta])
    # toutes les boîtes enregistrées
    B20 = {}
    for f in sorted((Path(args.out_dir) / "rect").glob("train_*.npz")):
        z = np.load(f)
        for n, b in zip(z["names"].tolist(), z["boites"]):
            B20[n] = b
    n = len(D.noms)
    P = np.zeros(D.C.shape, bool); regime = np.empty(n, object); params = {}
    for m in METHODES:
        for k in range(K):
            tr = (D.meths == m) & (D.pi != k); te = (D.meths == m) & (D.pi == k)
            p = regler(D.sous(tr), False); params[(m, k)] = p
            Dt = D.sous(te); P[te] = predire(Dt, p)
            f = Dt.sf >= p["tf"]; b = ~f & (Dt.sb >= p["tb"])
            regime[te] = np.where(f, "full", np.where(b, "boîte", "vide"))
    sc = miou_lot(P, D.G)
    print(f"H3b (repli {args.repli}) : {sc.mean():.4f} sur {n} images\n", flush=True)

    # 1. régimes
    print("1. Régime prédit contre régime vrai (n, mIoU moyen)")
    for m in METHODES:
        for fv in (False, True):
            s = (D.meths == m) & (full_vrai == fv)
            ligne = "  ".join(f"{r}: {int(((regime == r) & s).sum()):3d} ({sc[(regime == r) & s].mean() if ((regime == r) & s).any() else float('nan'):.3f})"
                              for r in ("full", "boîte", "vide"))
            print(f"  {m:10s} vrai {'full   ' if fv else 'partiel'} : {ligne}")
    # oracle régime parfait
    sc_reg = sc.copy()
    for i in range(n):
        m = D.meths[i]; p = params[(m, D.pi[i])]
        Di = D.sous(np.arange(n) == i)
        if full_vrai[i]:
            q = {**p, "tf": -1}                                       # forcé full
        else:
            q = {**p, "tf": 2}                                        # forcé partiel (boîte si sb >= tb)
        sc_reg[i] = miou_lot(predire(Di, q), D.G[i:i + 1])[0]
    print(f"  oracle régime parfait : {sc_reg.mean():.4f} ({sc_reg.mean() - sc.mean():+.4f})\n")

    # 2-3. boîtes
    part = ~full_vrai
    best20 = np.zeros(n); vraie = np.zeros(n); choisie = np.zeros(n); rang = np.full(n, -1)
    for i in np.where(part)[0]:
        g = D.G[i]; bs = B20[D.noms[i]]
        ious = [iou1(masque_boite(b), g) for b in bs]
        best20[i] = max(ious); choisie[i] = ious[0]; rang[i] = int(np.argmax(ious))
        lab, nb = ndimage.label(g)
        ys, xs = np.where(g)
        bb = np.zeros_like(g); bb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        vraie[i] = miou_lot(bb[None], g[None])[0]
    print("2. Images vraiment partielles : IoU (classe tatouée) de la boîte choisie, de la meilleure des 20 enregistrées")
    for m in METHODES:
        s = part & (D.meths == m)
        print(f"  {m:10s}: boîte choisie IoU médiane {np.median(choisie[s]):.3f} (> 0,5 : {(choisie[s] > .5).mean():.2f})   "
              f"meilleure des 20 : médiane {np.median(best20[s]):.3f} (> 0,5 : {(best20[s] > .5).mean():.2f})   "
              f"rang de la meilleure = 0 : {(rang[s] == 0).mean():.2f}")
    # oracle re-classement : boîte = meilleure des 20 quand on est en régime boîte ou vide (image partielle)
    sc_rc = sc.copy()
    for i in np.where(part)[0]:
        bs = B20[D.noms[i]]; j = int(np.argmax([iou1(masque_boite(b), D.G[i]) for b in bs]))
        sc_rc[i] = max(sc[i], miou_lot(masque_boite(bs[j])[None], D.G[i:i + 1])[0]) if regime[i] != "full" else sc[i]
    print(f"  oracle re-classement des 20 boîtes (images partielles hors régime full) : {sc_rc.mean():.4f} ({sc_rc.mean() - sc.mean():+.4f})")
    sc_bv = sc.copy(); sc_bv[part] = vraie[part]
    print(f"  oracle boîte vraie sur toutes les images partielles : {sc_bv.mean():.4f} ({sc_bv.mean() - sc.mean():+.4f})")
    rect = np.array([D.G[i].sum() / max(1, (np.ptp(np.where(D.G[i])[0]) + 1) * (np.ptp(np.where(D.G[i])[1]) + 1))
                     if D.G[i].any() else 1 for i in range(n)])
    print(f"  mIoU de la boîte vraie elle-même (perte de forme) : médiane {np.median(vraie[part]):.3f} ; "
          f"part des partiels non rectangulaires (rectangularité < 0,8) : {(rect[part] < .8).mean():.2f}\n")

    # 4. partiels sans boîte
    frac = D.G.reshape(n, -1).mean(1)
    print("4. Images vraiment partielles, par taille de zone : part en régime « vide » et score")
    for lo, hi in ((0, .01), (.01, .03), (.03, .1), (.1, .3), (.3, 1.01)):
        s = part & (frac >= lo) & (frac < hi)
        if s.any():
            print(f"  zone {lo:.0%}–{hi:.0%} : n={s.sum():3d}  vide {(regime[s] == 'vide').mean():.2f}  boîte {(regime[s] == 'boîte').mean():.2f}"
                  f"  full {(regime[s] == 'full').mean():.2f}  mIoU {sc[s].mean():.3f}  (tout à 0 : {miou_lot(np.zeros_like(D.G[s]), D.G[s]).mean():.3f})")
    print()
    # 5. images full
    print("5. Images vraiment full : mIoU actuel, tout à 1, et borne si les trous étaient parfaits")
    for m in METHODES:
        s = full_vrai & (D.meths == m)
        print(f"  {m:10s}: actuel {sc[s].mean():.3f}   tout à 1 {miou_lot(np.ones_like(D.G[s]), D.G[s]).mean():.3f}   "
              f"régime prédit full {(regime[s] == 'full').mean():.2f}")
    sc_f1 = sc.copy(); sc_f1[full_vrai & (regime == "full")] = miou_lot(np.ones_like(D.G[full_vrai & (regime == 'full')]), D.G[full_vrai & (regime == 'full')])
    print(f"  si « tout à 1 » sur les images prédites full : {sc_f1.mean():.4f} ({sc_f1.mean() - sc.mean():+.4f})")
    sc_tf = sc.copy(); sc_tf[full_vrai] = 1.0
    print(f"  borne : trous parfaits sur toutes les images full : {sc_tf.mean():.4f} ({sc_tf.mean() - sc.mean():+.4f})")
    sc_tp = sc.copy(); sc_tp[part] = 1.0
    print(f"  borne : images partielles parfaites : {sc_tp.mean():.4f} ({sc_tp.mean() - sc.mean():+.4f})")


if __name__ == "__main__":
    main()
