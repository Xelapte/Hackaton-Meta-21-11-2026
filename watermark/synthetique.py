"""Tâche 1 : données synthétiques (image, masque, cartes) pour entraîner le U-Net à voir les petites zones.

Diagnostic (21/11, 17h30) : la part tatouée du train est bimodale (un tiers < 10 %, un tiers > 90 %) ;
deviner la classe dominante donne déjà ~0,45 et nos cartes par position ne voient presque pas la petite zone.
Les embedders étant fournis, on fabrique autant d'exemples que voulu, pour une méthode cible :
  - support : une image du train ou du test tatouée par une AUTRE méthode (donc propre pour la cible) ;
  - « trou »      : toute l'image tatouée, puis 1 à 3 formes d'objet remises à l'original (ou remplacées par le
                    contenu d'une autre image propre, comme un objet ajouté) ; masque = hors des formes ;
  - « objet »     : un objet découpé dans une autre image tatouée, collé au même endroit sur l'image propre ;
  - « rectangle » : un rectangle (rapport <= 16:1, 2:1 pour TrustMark) tatoué à sa propre taille, avec parfois
                    un trou dedans (extension, retouche).
Pour chaque exemple : masque 128 x 128 (règle officielle) et les mêmes cartes que --mode cartes, normalisées
avec mu et sd de cartes_feats.npz. Sortie : <out_dir>/synth/<méthode>_{images,masks,feats}.npy.

  bash slurm/soumettre.sh 2 wm_synth_ps --venv watermark python watermark/synthetique.py --methode pixelseal --n 2000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.baseline import cartes_image  # noqa: E402

RAPPORT_MAX = {"pixelseal": 16, "trustmark": 2, "stegastamp": 16}


def reduire128(m512):
    """Règle officielle : une case 4 x 4 vaut 1 si au moins la moitié de ses pixels valent 1."""
    return (m512.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)


def forme_objet(rng, part):
    """Masque 512 x 512 d'une forme d'objet (polygone étoilé et/ou ellipses) couvrant environ `part` de l'image."""
    aire = part * 512 * 512
    r = np.sqrt(aire / np.pi)
    cy, cx = rng.uniform(r * 0.5, 512 - r * 0.5, 2) if r < 256 else (256, 256)
    im = Image.new("L", (512, 512), 0); d = ImageDraw.Draw(im)
    if rng.random() < 0.6:                                         # polygone étoilé, allongé et tourné
        k = int(rng.integers(5, 14)); ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        rad = r * rng.uniform(0.6, 1.4, k); el = rng.uniform(0.4, 1.0); rot = rng.uniform(0, np.pi)
        x = rad * np.cos(ang); y = rad * np.sin(ang) * el
        pts = [(cx + a * np.cos(rot) - b * np.sin(rot), cy + a * np.sin(rot) + b * np.cos(rot)) for a, b in zip(x, y)]
        d.polygon(pts, fill=255)
    else:                                                          # 1 à 3 ellipses qui se chevauchent
        for _ in range(int(rng.integers(1, 4))):
            ry, rx = r * rng.uniform(0.4, 1.2, 2); oy, ox = rng.normal(0, r * 0.4, 2)
            d.ellipse([cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry], fill=255)
    return np.asarray(im) > 127


def rectangle(rng, meth, part_min=0.05, part_max=0.9):
    """(y0, y1, x0, x1) d'un rectangle de rapport <= RAPPORT_MAX[meth], de côté >= 48 px."""
    for _ in range(100):
        part = np.exp(rng.uniform(np.log(part_min), np.log(part_max)))
        ratio = np.exp(rng.uniform(0, np.log(RAPPORT_MAX[meth])))
        if rng.random() < 0.5:
            ratio = 1 / ratio
        h = int(np.sqrt(part * 512 * 512 / ratio)); w = int(h * ratio)
        if 48 <= h <= 512 and 48 <= w <= 512:
            y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1))
            return y0, y0 + h, x0, x0 + w
    return 128, 384, 128, 384


def exemple(wm, meth, supports, rng, p_types):
    """Un exemple synthétique : (image uint8 512 x 512 x 3, masque 512 x 512 bool, message, type)."""
    nbits = wm.nbits
    msg = rng.integers(0, 2, nbits).astype(np.uint8)
    c = supports[int(rng.integers(len(supports)))]
    typ = rng.choice(["trou", "objet", "rectangle"], p=p_types)
    if typ == "trou":
        w = wm.embed(c, msg); img = w.copy(); trous = np.zeros((512, 512), bool)
        for _ in range(int(rng.integers(1, 4))):
            trous |= forme_objet(rng, np.exp(rng.uniform(np.log(0.002), np.log(0.3))))
        rempl = c if rng.random() < 0.6 else supports[int(rng.integers(len(supports)))]   # restauré / objet ajouté
        img[trous] = rempl[trous]; masque = ~trous
    elif typ == "objet":
        c2 = supports[int(rng.integers(len(supports)))]
        w2 = wm.embed(c2, msg)
        obj = forme_objet(rng, np.exp(rng.uniform(np.log(0.002), np.log(0.3))))
        img = c.copy(); img[obj] = w2[obj]; masque = obj
    else:
        y0, y1, x0, x1 = rectangle(rng, meth)
        img = c.copy(); img[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(c[y0:y1, x0:x1]), msg)
        masque = np.zeros((512, 512), bool); masque[y0:y1, x0:x1] = True
        if rng.random() < 0.5:                                    # retouche dans le rectangle
            trou = forme_objet(rng, np.exp(rng.uniform(np.log(0.002), np.log(0.1)))) & masque
            img[trou] = c[trou]; masque &= ~trou
    return img, masque, msg, typ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--methode", choices=["pixelseal", "trustmark", "stegastamp"], required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tache", type=Path, default=off)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--feats", default=str(off / "output" / "cartes_feats.npz"), help="pour mu, sd et les canaux")
    ap.add_argument("--p_types", default="0.4,0.3,0.3", help="probabilités trou, objet, rectangle")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed + 1000 * ["pixelseal", "trustmark", "stegastamp"].index(args.methode))
    sys.path.insert(0, str(args.tache))
    from wmloc.watermarks import load_watermark

    z = np.load(args.feats)
    canaux = z["canaux"].tolist(); mu, sd = z["mu"], z["sd"]
    # mêmes listes de canaux que --mode cartes
    args.feat_ps = ",".join(c[3:] for c in canaux if c.startswith("ps_"))
    args.feat_tm = ",".join(c[3:] for c in canaux if c.startswith("tm_"))
    args.feat_ss = ",".join(c[3:] for c in canaux if c.startswith("ss_"))

    supports = []
    for split in ("train", "test"):
        for m in json.loads((args.data / split / "metadata.json").read_text()):
            if m["method"] != args.methode:
                supports.append(np.asarray(Image.open(args.data / split / "images" / f"{m['image_name']}.png").convert("RGB")))
    print(f"{args.methode} : {len(supports)} images support (tatouées par une autre méthode)")

    wm = load_watermark(args.methode, models_dir=args.tache / "models", device=dev)
    p_types = [float(v) for v in args.p_types.split(",")]
    out = Path(args.out_dir) / "synth"; out.mkdir(parents=True, exist_ok=True)
    pre = out / args.methode
    images = np.lib.format.open_memmap(f"{pre}_images.npy", "w+", np.uint8, (args.n, 512, 512, 3))
    masks = np.zeros((args.n, 128, 128), np.uint8)
    feats = np.zeros((args.n, len(canaux), 128, 128), np.float16)
    types, t0 = [], time.time()
    for i in range(args.n):
        img, m512, msg, typ = exemple(wm, args.methode, supports, rng, p_types)
        d, cs = cartes_image(wm, args.methode, img, msg, args, dev)
        f = np.zeros((len(canaux), 128, 128), np.float32)
        f[d:d + len(cs)] = (np.stack(cs) - mu[d:d + len(cs), None, None]) / sd[d:d + len(cs), None, None]
        images[i] = img; masks[i] = reduire128(m512.astype(np.float32)); feats[i] = f; types.append(typ)
        if i % 200 == 0 or i == args.n - 1:
            print(f"  {i + 1}/{args.n}  ({time.time() - t0:.0f} s)  part tatouée moyenne {masks[:i + 1].mean():.3f}")
    images.flush()
    np.save(f"{pre}_masks.npy", masks); np.save(f"{pre}_feats.npy", feats)
    np.save(f"{pre}_types.npy", np.array(types))
    print(f"Écrit : {pre}_{{images,masks,feats,types}}.npy ({args.n} exemples, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
