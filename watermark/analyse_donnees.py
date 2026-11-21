"""Tâche 1 : analyse des données du train (CPU, léger) et visualisation d'exemples en PNG.

Par méthode x full_watermark : part tatouée, rapport largeur/hauteur de la boîte englobante de la zone
tatouée (et de la zone propre), nombre de composantes connexes, « rectangularité » (part de la boîte
englobante qui est tatouée). Écrit aussi --n_png exemples (image | masque | carte optionnelle) dans --out_dir.

  python watermark/analyse_donnees.py --out_dir $OFFICIEL/Hackathon_Watermark_Localisation_Task/output/recherche
  python watermark/analyse_donnees.py --fake
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

METHODES = ["pixelseal", "trustmark", "stegastamp"]


def stats_masque(m):
    """m : 128 x 128 (1 = tatoué)."""
    out = {"part": float(m.mean())}
    for nom, z in (("tat", m == 1), ("propre", m == 0)):
        lab, n = ndimage.label(z)
        out[f"ncomp_{nom}"] = int(n)
        if n:
            ys, xs = np.where(z)
            h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
            out[f"rapport_{nom}"] = float(max(h, w) / min(h, w))
            out[f"rect_{nom}"] = float(z.sum() / (h * w))          # 1 = rectangle plein
            out[f"boite_{nom}"] = float(h * w / m.size)
    return out


def q(v, qs=(0.1, 0.5, 0.9)):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return "/".join(f"{x:.2f}" for x in np.quantile(v, qs)) if v else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", type=Path, default=off / "output" / "recherche")
    ap.add_argument("--n_png", type=int, default=24)
    ap.add_argument("--cartes", default=None, help="cartes_feats.npz : ajoute une carte à côté de chaque exemple")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    if args.fake:
        meta = [{"image_name": f"img{i}", "method": METHODES[i % 3], "full_watermark": bool(i % 2)} for i in range(12)]
        M = {}
        for m in meta:
            k = np.zeros((128, 128), np.uint8); y, x = rng.integers(0, 64, 2); k[y:y + 40, x:x + 60] = 1; M[m["image_name"]] = k
        args.out_dir = Path("/tmp/wm_fake_analyse"); args.n_png = 3
    else:
        meta = json.loads((args.data / "train" / "metadata.json").read_text())
        z = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    S = {m["image_name"]: stats_masque(M[m["image_name"]]) for m in meta}
    print(f"{'groupe':24s} {'n':>4s}  part q10/50/90     ncomp tat  ncomp propre  rapport tat  rect tat   rapport propre")
    for meth in METHODES:
        for full in (False, True):
            g = [S[m["image_name"]] for m in meta if m["method"] == meth and bool(m["full_watermark"]) == full]
            print(f"{meth + (' full' if full else ' partiel'):24s} {len(g):4d}  {q([s['part'] for s in g]):16s}  "
                  f"{q([s['ncomp_tat'] for s in g]):10s} {q([s['ncomp_propre'] for s in g]):12s}  "
                  f"{q([s.get('rapport_tat') for s in g]):11s} {q([s.get('rect_tat') for s in g]):9s}  "
                  f"{q([s.get('rapport_propre') for s in g])}", flush=True)
    # bimodalité
    parts = np.array([S[m["image_name"]]["part"] for m in meta])
    print(f"\npart tatouée : < 5 % {np.mean(parts < .05):.2f}, < 10 % {np.mean(parts < .1):.2f}, "
          f"10-90 % {np.mean((parts >= .1) & (parts <= .9)):.2f}, > 90 % {np.mean(parts > .9):.2f}, "
          f"> 95 % {np.mean(parts > .95):.2f}", flush=True)
    for full in (False, True):
        p = np.array([S[m["image_name"]]["part"] for m in meta if bool(m["full_watermark"]) == full])
        print(f"  full_watermark={full}: part q10/50/90 {q(p)} ; < 10 % {np.mean(p < .1):.2f} ; > 90 % {np.mean(p > .9):.2f}")
    (args.out_dir / "stats_masques.json").write_text(json.dumps(S))

    # exemples PNG : par méthode, des parts tatouées variées
    cartes = None
    if args.cartes:
        zc = np.load(args.cartes); cartes = (dict(zip(zc["names"].tolist(), zc["feats"])), zc["canaux"].tolist())
    choix = []
    for meth in METHODES:
        its = sorted([m for m in meta if m["method"] == meth], key=lambda m: S[m["image_name"]]["part"])
        idx = np.linspace(0, len(its) - 1, max(1, args.n_png // 3)).astype(int)
        choix += [its[i] for i in idx]
    for m in choix:
        n = m["image_name"]
        img = (np.asarray(Image.open(args.data / "train" / "images" / f"{n}.png").convert("RGB")) if not args.fake
               else rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
        msk = np.kron(M[n], np.ones((4, 4), np.uint8)) * 255
        panneaux = [img, np.stack([msk] * 3, -1)]
        if cartes is not None and n in cartes[0]:
            j = {"pixelseal": "ps_512", "trustmark": "tm_224", "stegastamp": "ss_16"}[m["method"]]
            c = cartes[0][n][cartes[1].index(j)].astype(np.float32)
            c = np.clip((c + 2) / 4, 0, 1); c = (np.kron(c, np.ones((4, 4))) * 255).astype(np.uint8)
            panneaux.append(np.stack([c] * 3, -1))
        sep = np.full((512, 8, 3), 128, np.uint8)
        out = np.concatenate(sum([[p, sep] for p in panneaux], [])[:-1], 1)
        nom = f"{m['method']}_{'full' if m['full_watermark'] else 'partiel'}_{S[n]['part']:.3f}_{n}.png"
        Image.fromarray(out).save(args.out_dir / nom)
    print(f"\n{len(choix)} exemples PNG (image | masque{' | carte' if cartes else ''}) dans {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
