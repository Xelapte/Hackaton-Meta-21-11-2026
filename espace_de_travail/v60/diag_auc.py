"""v60 / diagnostic (CPU leger) : dans les images 'full' (trous), quel signal existant separe trou / tatoue ?
AUC par image (cases 128x128), moyenne sur les images full d'une methode. Idem images partielles."""
import json, sys
from pathlib import Path
import numpy as np
T = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
meta = {m["image_name"]: m for m in json.loads((T / "release/train/metadata.json").read_text())}
z = np.load(T / "release/train/masks_128.npz"); G = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))

def auc(s, y):
    s = s.ravel().astype(np.float64); y = y.ravel()
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable") + 1.0
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def bilan(nom, cartes, meth):
    for full in (True, False):
        a = [auc(cartes[n], G[n]) for n in cartes if meta[n]["method"] == meth and bool(meta[n]["full_watermark"]) == full]
        a = np.array([v for v in a if not np.isnan(v)])
        if len(a): print(f"  {meth:10s} {nom:14s} full={full!s:5s} n={len(a):3d}  AUC moy {a.mean():.3f}  med {np.median(a):.3f}  q25 {np.quantile(a,.25):.3f} q75 {np.quantile(a,.75):.3f}", flush=True)

zc = np.load(T / "output/cartes_feats.npz"); can = zc["canaux"].tolist(); print("canaux", can, "mu", zc["mu"], "sd", zc["sd"])
noms = zc["names"].tolist(); F = zc["feats"]
for meth in ("pixelseal", "trustmark", "stegastamp"):
    for j, c in enumerate(can):
        if c[:2] != meth[:1] + {"pixelseal": "s", "trustmark": "m", "stegastamp": "s"}[meth]: continue
        if (meth == "pixelseal") != c.startswith("ps") and meth != "stegastamp": continue
        if meth == "stegastamp" and not c.startswith("ss"): continue
        if meth == "pixelseal" and not c.startswith("ps"): continue
        bilan(c, {n: F[i, j] for i, n in enumerate(noms) if n in meta}, meth)
zo = np.load(T / "output/unet_h5a_feats_oof.npz")
U = dict(zip(zo["names"].tolist(), zo["probs"]))
for meth in ("pixelseal", "trustmark", "stegastamp"): bilan("unet_oof", U, meth)
for meth in ("pixelseal", "trustmark", "stegastamp"):
    zr = np.load(T / f"output/reembed_{meth}.npz")
    for b in ("16", "32", "64"):
        bilan(f"reembed_{b}", {k.split("/")[1]: zr[k] for k in zr.files if k.startswith(b + "/")}, meth)
zd = np.load(T / "output/direct_maps.npz"); Dm = {n: m for n, m in zip(zd["names"].tolist(), zd["maps"]) if n in meta}
zw = np.load(T / "output/window_maps.npz"); Wm = {n: m for n, m in zip(zw["names"].tolist(), zw["maps"]) if n in meta}
for meth in ("pixelseal", "trustmark", "stegastamp"): bilan("direct", Dm, meth); bilan("window", Wm, meth)
