"""Tâche 1, H8b : recherche FINE du rectangle tatoué (petits rectangles PixelSeal et StegaStamp).

Constat (H8a) : sur les images partielles rectangulaires, une recherche locale fine autour de la vraie boîte
(pas 1/16) dépasse le niveau du hasard pour 95 % des StegaStamp et 66 % des PixelSeal, alors que la grille de
rectangles.py (pas d'une demi-taille) n'en trouve que 5 % / 33 % : c'est un problème d'alignement.
Grille fixée d'avance (coût, pas de réglage sur le score) :
  - tailles en progression géométrique (--facteur) depuis --t_min, rapport <= 16:1 (2:1 TrustMark),
    pas = --pas x taille (au plus 1/3) ;
  - raffinement autour des --top meilleures : décalages -1/8..1/8 par 1/16, échelles 0,85 / 1 / 1,15.
Sortie : même format que rectangles.py, dans <out_dir>/rect_fin/ ; la règle (regle.py --rect_dir) prend la
meilleure boîte des deux recherches.

  bash slurm/soumettre.sh 2 wm_fin_ss --venv watermark python watermark/recherche_fine.py --methode stegastamp \\
      --facteur 1.5 --pas 0.33 --top 20
  python watermark/recherche_fine.py --methode stegastamp --fake
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.rectangles import FauxTatouage, RAPPORT_MAX, precision_boites  # noqa: E402


def tailles(t_min, facteur):
    t = sorted({int(round(t_min * facteur ** i)) for i in range(40) if t_min * facteur ** i <= 512})
    return t + ([512] if t[-1] != 512 else [])


def grille_fine(meth, T, pas):
    boites = []
    for h in T:
        for w in T:
            if max(h, w) / min(h, w) > RAPPORT_MAX[meth]:
                continue
            sy, sx = max(8, int(h * pas)), max(8, int(w * pas))
            ys = sorted(set(list(range(0, 512 - h + 1, sy)) + [512 - h]))
            xs = sorted(set(list(range(0, 512 - w + 1, sx)) + [512 - w]))
            boites += [(y, y + h, x, x + w) for y in ys for x in xs]
    return boites


def raffiner_fin(b, meth):
    y0, y1, x0, x1 = b; h, w = y1 - y0, x1 - x0; out = set()
    for eh in (0.85, 1.0, 1.15):
        for ew in (0.85, 1.0, 1.15):
            hh, ww = int(min(512, max(24, h * eh))), int(min(512, max(24, w * ew)))
            if max(hh, ww) / min(hh, ww) > RAPPORT_MAX[meth]:
                continue
            for dy in np.arange(-2, 3) / 16:
                for dx in np.arange(-2, 3) / 16:
                    cy = (y0 + y1) / 2 + dy * h; cx = (x0 + x1) / 2 + dx * w
                    ny = int(np.clip(cy - hh / 2, 0, 512 - hh)); nx = int(np.clip(cx - ww / 2, 0, 512 - ww))
                    out.add((ny, ny + hh, nx, nx + ww))
    return list(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--methode", choices=["pixelseal", "trustmark", "stegastamp"], required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--parts", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--t_min", type=int, default=48)
    ap.add_argument("--facteur", type=float, default=1.41)
    ap.add_argument("--pas", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--garder", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--tache", type=Path, default=off)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    if args.fake:
        wm = FauxTatouage(); meta = [{"image_name": f"i{k}", "method": args.methode,
                                       "message": "".join(rng.choice(["0", "1"], 100))} for k in range(2)]
        charger = lambda n: rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        out = Path("/tmp/wm_fake_rect_fin"); args.t_min, args.facteur = 128, 2.0
    else:
        sys.path.insert(0, str(args.tache))
        from wmloc.watermarks import load_watermark
        wm = load_watermark(args.methode, models_dir=args.tache / "models", device=dev)
        meta = [m for m in json.loads((args.data / args.split / "metadata.json").read_text()) if m["method"] == args.methode]
        charger = lambda n: np.asarray(Image.open(args.data / args.split / "images" / f"{n}.png").convert("RGB"))
        out = Path(args.out_dir)
    if args.parts > 1:
        meta = meta[args.part::args.parts]
    if args.max_images:
        meta = meta[:args.max_images]
    (out / "rect_fin").mkdir(parents=True, exist_ok=True)
    T = tailles(args.t_min, args.facteur); g = grille_fine(args.methode, T, args.pas)
    print(f"{args.methode} / {args.split} : {len(meta)} images ; tailles {T} ; {len(g)} boîtes grossières par image",
          flush=True)
    noms, sortie, full = [], [], []
    t0 = time.time()
    for n, m in enumerate(meta):
        x = torch.from_numpy(charger(m["image_name"])).permute(2, 0, 1).float().div(255).to(dev)
        signe = torch.tensor([int(c) * 2 - 1 for c in m["message"]][:wm.nbits], dtype=torch.float32, device=dev)
        p = precision_boites(wm, x, g, signe, args.bs)
        top = [g[i] for i in np.argsort(-p)[:args.top]]
        fin = list({b for t in top for b in raffiner_fin(t, args.methode)})
        pf = precision_boites(wm, x, fin, signe, args.bs)
        tout = g + fin; pt = np.concatenate([p, pf])
        garde, vus = [], set()
        for i in np.argsort(-pt):
            if tout[i] not in vus:
                vus.add(tout[i]); garde.append((*tout[i], pt[i]))
            if len(garde) >= args.garder:
                break
        noms.append(m["image_name"]); sortie.append(np.array(garde, np.float32))
        full.append(float(precision_boites(wm, x, [(0, 512, 0, 512)], signe, 1)[0]))
        if n % 10 == 0 or n == len(meta) - 1:
            print(f"  {n + 1}/{len(meta)}  meilleure {garde[0][4]:.3f}  ({time.time() - t0:.0f} s, "
                  f"{(time.time() - t0) / (n + 1):.1f} s / image)", flush=True)
    suffixe = f"_{args.part}" if args.parts > 1 else ""
    chemin = out / "rect_fin" / f"{args.split}_{args.methode}{suffixe}.npz"
    np.savez(chemin, names=np.array(noms), boites=np.stack(sortie), full=np.array(full, np.float32))
    print(f"Écrit : {chemin}", flush=True)


if __name__ == "__main__":
    main()
