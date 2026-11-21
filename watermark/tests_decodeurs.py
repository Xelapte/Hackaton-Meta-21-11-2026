"""Tâche 1 : deux tests rapides des décodeurs (avant de construire dessus).

  --test oracle      (H3a) Images partielles du train : précision de bits (message connu) en décodant la boîte
                     englobante VRAIE de la zone tatouée, contre des boîtes décalées, agrandies, réduites,
                     tirées au hasard, et l'image entière. Si la vraie boîte se détache nettement, une recherche
                     de rectangle est viable. Mesure aussi la vitesse de décodage (boîtes / s).
  --test occlusion   (H6a) Images full du train : on floute une boîte et on mesure la variation relative de la
                     marge de décodage de l'image entière (moyenne de logit x signe du bit). Boîte d'un vrai trou
                     (attendu ~0) contre boîte de même taille en zone tatouée (attendu < 0). Si ça sépare, on peut
                     chercher les trous par occlusion.
  --test invariance  (H4) Images full du train : précision de bits de l'image, de ses miroirs et rotations.
                     Décide si l'augmentation miroir / rotation est permise.

  bash slurm/soumettre.sh 2 wm_h3a --venv watermark python watermark/tests_decodeurs.py --test oracle
  bash slurm/soumettre.sh 3 wm_h4  --venv watermark python watermark/tests_decodeurs.py --test invariance
  python watermark/tests_decodeurs.py --test oracle --fake
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
from scipy import ndimage

METHODES = ["pixelseal", "trustmark", "stegastamp"]


class FauxTatouage:
    nbits = 100
    def logits(self, x):
        return torch.randn(len(x), self.nbits)


def boite_principale(m128):
    """Boîte englobante (y0, y1, x0, x1, en pixels 512) de la plus grande composante tatouée."""
    lab, n = ndimage.label(m128)
    if n == 0:
        return None
    k = 1 + int(np.argmax(ndimage.sum(m128, lab, range(1, n + 1))))
    ys, xs = np.where(lab == k)
    return ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4


def borner(y0, y1, x0, x1, mini=16):
    h = max(mini, y1 - y0); w = max(mini, x1 - x0)
    h = min(h, 512); w = min(w, 512)
    y0 = int(np.clip(y0, 0, 512 - h)); x0 = int(np.clip(x0, 0, 512 - w))
    return y0, y0 + h, x0, x0 + w


def variantes(b, rng):
    y0, y1, x0, x1 = b; h, w = y1 - y0, x1 - x0; cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    v = {"vraie": borner(y0, y1, x0, x1),
         "décalée 25 %": borner(y0 + h // 4, y1 + h // 4, x0 + w // 4, x1 + w // 4),
         "décalée 50 %": borner(y0, y1, x0 + w // 2, x1 + w // 2),
         "agrandie x2": borner(int(cy - h), int(cy + h), int(cx - w), int(cx + w)),
         "réduite x0,5": borner(int(cy - h / 4), int(cy + h / 4), int(cx - w / 4), int(cx + w / 4)),
         "image entière": (0, 512, 0, 512)}
    for _ in range(50):                                   # même taille, ailleurs, peu de recouvrement
        ry = int(rng.integers(0, 512 - min(h, 512) + 1)); rx = int(rng.integers(0, 512 - min(w, 512) + 1))
        r = borner(ry, ry + h, rx, rx + w)
        inter = max(0, min(r[1], y1) - max(r[0], y0)) * max(0, min(r[3], x1) - max(r[2], x0))
        if inter < 0.1 * h * w:
            v["hasard"] = r; break
    return v


def bits_justes(wm, crops, msg, dev):
    x = torch.stack([torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float().div(255) for c in crops])
    lg = wm.logits(x.to(dev) if not isinstance(wm, FauxTatouage) else x).float().cpu()
    return ((lg > 0).int().numpy() == msg[None].astype(int)).mean(1)


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--test", choices=["oracle", "invariance", "occlusion"], required=True)
    ap.add_argument("--flou", type=float, default=4.0, help="occlusion : rayon du flou gaussien (pixels)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tache", type=Path, default=off)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--n", type=int, default=40, help="images par méthode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    if args.fake:
        meta = [{"image_name": f"i{k}", "method": METHODES[k % 3], "full_watermark": bool(k % 2),
                 "message": "".join(rng.choice(["0", "1"], 100))} for k in range(12)]
        M = {}
        for m in meta:
            k = np.zeros((128, 128), np.uint8); k[20:50, 30:90] = 1; M[m["image_name"]] = 1 - k if m["full_watermark"] else k
        charger = lambda n: rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        args.n = 2
    else:
        sys.path.insert(0, str(args.tache))
        meta = json.loads((args.data / "train" / "metadata.json").read_text())
        z = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
        charger = lambda n: np.asarray(Image.open(args.data / "train" / "images" / f"{n}.png").convert("RGB"))
    for meth in METHODES:
        if args.fake:
            wm = FauxTatouage()
        else:
            from wmloc.watermarks import load_watermark
            wm = load_watermark(meth, models_dir=args.tache / "models", device=dev)
            if meth == "stegastamp":
                print(f"stegastamp : fournisseurs ONNX {wm.sess.get_providers()}", flush=True)
        full = args.test in ("invariance", "occlusion")
        its = [m for m in meta if m["method"] == meth and bool(m["full_watermark"]) == full]
        its = [its[i] for i in rng.permutation(len(its))[:args.n]]
        res = {}; t0 = time.time(); ncrops = 0
        for m in its:
            img = charger(m["image_name"]); msg = np.array([int(c) for c in m["message"]][:getattr(wm, "nbits", 100)])
            if args.test == "oracle":
                b = boite_principale(M[m["image_name"]])
                if b is None:
                    continue
                v = variantes(b, rng)
                crops = [img[y0:y1, x0:x1] for (y0, y1, x0, x1) in v.values()]
                for nom, (y0, y1, x0, x1), c in zip(v, v.values(), crops):
                    res.setdefault(nom, []).append(float(bits_justes(wm, [c], msg, dev)[0]))
                res.setdefault("_taille", []).append((b[1] - b[0]) * (b[3] - b[2]) / 512 ** 2)
                ncrops += len(crops)
            elif args.test == "occlusion":
                from PIL import ImageFilter
                g = M[m["image_name"]]
                lab, nb = ndimage.label(g == 0)
                if nb == 0:
                    continue
                kk = 1 + int(np.argmax(ndimage.sum(g == 0, lab, range(1, nb + 1))))
                ys, xs = np.where(lab == kk)
                b = (ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4)
                h, w = b[1] - b[0], b[3] - b[2]
                tat = None                                   # même taille, entièrement (>= 95 %) en zone tatouée
                for _ in range(200):
                    ry = int(rng.integers(0, 512 - h + 1)); rx = int(rng.integers(0, 512 - w + 1))
                    if g[ry // 4:(ry + h) // 4, rx // 4:(rx + w) // 4].mean() >= 0.95:
                        tat = (ry, ry + h, rx, rx + w); break
                if tat is None:
                    continue
                flou = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(args.flou)))
                def occ(bb):
                    o = img.copy(); o[bb[0]:bb[1], bb[2]:bb[3]] = flou[bb[0]:bb[1], bb[2]:bb[3]]; return o
                ims = [img, occ(b), occ(tat)]
                xx = torch.stack([torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float().div(255) for c in ims])
                lg = wm.logits(xx.to(dev) if not isinstance(wm, FauxTatouage) else xx).float().cpu().numpy()
                sg = msg.astype(np.float32) * 2 - 1
                marge = (lg * sg[None]).mean(1)
                res.setdefault("trou", []).append(float(marge[1] / marge[0] - 1))
                res.setdefault("zone tatouée", []).append(float(marge[2] / marge[0] - 1))
                res.setdefault("_taille", []).append(h * w / 512 ** 2)
                ncrops += 3
            else:
                x = {"original": img, "miroir horizontal": img[:, ::-1], "miroir vertical": img[::-1],
                     "rotation 90": np.rot90(img), "rotation 180": np.rot90(img, 2)}
                for nom, c in x.items():
                    res.setdefault(nom, []).append(float(bits_justes(wm, [c], msg, dev)[0]))
                ncrops += len(x)
        dt = time.time() - t0
        print(f"\n=== {meth} : {len(its)} images {'full' if full else 'partielles'} "
              f"({ncrops / max(dt, 1e-6):.1f} décodages / s, un par un)", flush=True)
        for nom, v in res.items():
            if nom.startswith("_"):
                continue
            v = np.array(v)
            lib = "variation relative de la marge" if args.test == "occlusion" else "bits justes"
            print(f"  {nom:18s}: {lib} moyenne {v.mean():.3f}  q10/50/90 {np.quantile(v, [.1, .5, .9]).round(3)}", flush=True)
        if args.test == "occlusion" and "trou" in res:
            print(f"  AUC (trou > zone tatouée) : {auc(res['trou'], res['zone tatouée']):.3f} ; "
                  f"taille médiane de la boîte {np.median(res['_taille']):.3f}", flush=True)
        if args.test == "oracle" and "hasard" in res:
            vr = np.array(res["vraie"]); ts = np.array(res["_taille"])
            print(f"  AUC vraie boîte contre hasard : {auc(res['vraie'], res['hasard']):.3f} ; "
                  f"vraie > 0,75 : {(vr > .75).mean():.2f}", flush=True)
            for lo, hi in [(0, .01), (.01, .04), (.04, .15), (.15, 1.01)]:
                k = (ts >= lo) & (ts < hi)
                if k.any():
                    print(f"    boîte de {lo:.0%} à {hi:.0%} de l'image (n={k.sum()}) : vraie {vr[k].mean():.3f}", flush=True)
        if not args.fake:
            del wm; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
