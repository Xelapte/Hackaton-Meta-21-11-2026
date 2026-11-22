#!/usr/bin/env python
"""v34 - Prepare les images exterieures : lit les JPEG directement DANS l'archive zip (aucune decompression sur disque),
redimensionne (petit cote 224 + recadrage central) et enregistre UN seul fichier uint8 [M,224,224,3].
Usage : python prep.py <archive.zip> <sortie.npy> [nb_max]"""
import sys, zipfile, io, numpy as np
from PIL import Image
src, out = sys.argv[1], sys.argv[2]; nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
z = zipfile.ZipFile(src); names = sorted(n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))[:nmax]
arr = np.zeros((len(names), 224, 224, 3), np.uint8)
for i, n in enumerate(names):
    im = Image.open(io.BytesIO(z.read(n))).convert("RGB"); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2; arr[i] = np.asarray(im.crop((l, t, l + 224, t + 224)))
    if i % 1000 == 0: print(i, flush=True)
import os
with open(out + ".tmp", "wb") as f: np.save(f, arr)
os.rename(out + ".tmp", out); print("saved", out, arr.shape, flush=True)
