#!/usr/bin/env python
"""v38 - Prepare des images GENEREES PAR IA (DiffusionDB, public) : lit les images dans plusieurs archives zip sans les
decompresser, redimensionne en 224x224 et enregistre UN seul fichier. Usage : python prep_multi.py <sortie.npy> <zip1> <zip2> ..."""
import sys, os, zipfile, io, numpy as np
from PIL import Image
out, zips = sys.argv[1], sys.argv[2:]; imgs = []
for zp in zips:
    z = zipfile.ZipFile(zp)
    for n in sorted(x for x in z.namelist() if x.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        try:
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        except Exception:
            continue
        w, h = im.size; s = 224 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2; imgs.append(np.asarray(im.crop((l, t, l + 224, t + 224))))
    print(zp, len(imgs), flush=True)
arr = np.stack(imgs)
with open(out + ".tmp", "wb") as f: np.save(f, arr)
os.rename(out + ".tmp", out); print("saved", out, arr.shape, flush=True)
