"""Tâche 1 : données synthétiques v2, fidèles aux quatre retouches de l'énoncé officiel, pour le détecteur appris.

Pour une méthode cible m, chaque exemple part d'une image support A du TRAIN tatouée par une AUTRE méthode
(donc propre pour m ; aucune image du test n'est utilisée), avec un recadrage-agrandissement et un miroir
appliqués AVANT le tatouage (le tatouage, lui, n'est jamais retourné : H4). Types (proportions fixées d'après le train) :
  - outpainting (30 %)    : un rectangle (rapport <= 16:1, 2:1 TrustMark) de A tatoué à sa propre taille,
                            le reste de A intact ; masque = rectangle ;
  - transplantation (20 %): un objet découpé dans une autre image C tatouée en entier, collé au même endroit ;
  - retrait (25 %)        : A tatouée en entier, 1 à 3 trous en forme d'objet rebouchés par inpainting OpenCV
                            (+ bruit) ou par le contenu d'une autre image propre ;
  - ajout (25 %)          : A tatouée en entier, un objet d'une autre image propre collé par-dessus.
Les messages sont aléatoires (le détecteur n'utilise pas le message). Masques 128 x 128 (règle officielle).
Sortie : <out_dir>/synth_v2/<méthode>_{images,masks,types}.npy

  bash slurm/soumettre.sh 2 wm_sv2_ss --venv watermark python watermark/synth_v2.py --methode stegastamp --n 12000
  python watermark/synth_v2.py --methode trustmark --fake
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.synthetique import forme_objet, rectangle, reduire128  # noqa: E402

TYPES = ["outpainting", "transplantation", "retrait", "ajout"]
P_TYPES = [0.30, 0.20, 0.25, 0.25]


class FauxTatouage:
    nbits = 100
    def embed(self, img, bits):
        return np.clip(img.astype(np.int16) + 3, 0, 255).astype(np.uint8)


def support(supports, rng):
    """Une image support recadrée-agrandie (échelle 0,6–1) et éventuellement retournée, en 512 x 512."""
    a = supports[int(rng.integers(len(supports)))]
    s = rng.uniform(0.6, 1.0); t = int(512 * s)
    y, x = rng.integers(0, 512 - t + 1, 2)
    a = a[y:y + t, x:x + t]
    if rng.random() < 0.5:
        a = a[:, ::-1]
    if t != 512:
        a = cv2.resize(np.ascontiguousarray(a), (512, 512), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(a)


def part_objet(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def exemple(wm, meth, supports, rng):
    msg = rng.integers(0, 2, wm.nbits).astype(np.uint8)
    a = support(supports, rng)
    typ = TYPES[int(rng.choice(len(TYPES), p=P_TYPES))]
    if typ == "outpainting":
        y0, y1, x0, x1 = rectangle(rng, meth, part_min=0.004, part_max=0.6)
        img = a.copy(); img[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(a[y0:y1, x0:x1]), msg)
        m = np.zeros((512, 512), bool); m[y0:y1, x0:x1] = True
    elif typ == "transplantation":
        c = support(supports, rng); wc = wm.embed(c, msg)
        obj = forme_objet(rng, part_objet(rng, 0.003, 0.2))
        img = a.copy(); img[obj] = wc[obj]; m = obj
    elif typ == "retrait":
        w = wm.embed(a, msg); trous = np.zeros((512, 512), bool)
        for _ in range(int(rng.integers(1, 4))):
            trous |= forme_objet(rng, part_objet(rng, 0.003, 0.15))
        if rng.random() < 0.5:
            r = cv2.inpaint(w, trous.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
            r += rng.normal(0, rng.uniform(1, 4), r.shape)                 # grain, pour ne pas « lisser » la réponse
            img = w.copy(); img[trous] = np.clip(r, 0, 255).astype(np.uint8)[trous]
        else:
            d = support(supports, rng); img = w.copy(); img[trous] = d[trous]
        m = ~trous
    else:                                                                    # ajout
        w = wm.embed(a, msg); d = support(supports, rng)
        obj = forme_objet(rng, part_objet(rng, 0.003, 0.25))
        img = w.copy(); img[obj] = d[obj]; m = ~obj
    return img, m, typ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--methode", choices=["pixelseal", "trustmark", "stegastamp"], required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tache", type=Path, default=off)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed + 1000 * ["pixelseal", "trustmark", "stegastamp"].index(args.methode))
    if args.fake:
        wm = FauxTatouage(); supports = [rng.integers(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(4)]
        args.n = 6; out = Path("/tmp/wm_fake_sv2")
    else:
        sys.path.insert(0, str(args.tache))
        from wmloc.watermarks import load_watermark
        wm = load_watermark(args.methode, models_dir=args.tache / "models", device=dev)
        supports = [np.asarray(Image.open(args.data / "train" / "images" / f"{m['image_name']}.png").convert("RGB"))
                    for m in json.loads((args.data / "train" / "metadata.json").read_text()) if m["method"] != args.methode]
        out = Path(args.out_dir)
    print(f"{args.methode} : {len(supports)} supports (train, autres méthodes), {args.n} exemples", flush=True)
    (out / "synth_v2").mkdir(parents=True, exist_ok=True)
    pre = out / "synth_v2" / args.methode
    images = np.lib.format.open_memmap(f"{pre}_images.npy", "w+", np.uint8, (args.n, 512, 512, 3))
    masks = np.zeros((args.n, 128, 128), np.uint8); types = []
    t0 = time.time()
    for i in range(args.n):
        img, m, typ = exemple(wm, args.methode, supports, rng)
        images[i] = img; masks[i] = reduire128(m.astype(np.float32)); types.append(typ)
        if i % 500 == 0 or i == args.n - 1:
            print(f"  {i + 1}/{args.n}  ({time.time() - t0:.0f} s)  part tatouée moyenne {masks[:i + 1].mean():.3f}", flush=True)
    images.flush(); np.save(f"{pre}_masks.npy", masks); np.save(f"{pre}_types.npy", np.array(types))
    fr = masks.reshape(len(masks), -1).mean(1)
    print(f"Écrit : {pre}_*.npy ; part tatouée q10/50/90 {np.quantile(fr, [.1, .5, .9]).round(3)} ; "
          f"< 10 % {np.mean(fr < .1):.2f}, > 90 % {np.mean(fr > .9):.2f}", flush=True)


if __name__ == "__main__":
    main()
