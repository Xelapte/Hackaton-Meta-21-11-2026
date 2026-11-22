"""v60 / signal pixel « re-tatouer et correler » (indice 4 du keynote), pour les 3 methodes, train + test.

Idee simple : on connait le message. On re-tatoue l'image avec le VRAI message et avec un FAUX message
(aleatoire). La difference d = embed(I, vrai) - embed(I, faux) est le « motif attendu » du tatouage, debarrasse
de tout ce qui ne depend pas du message. La ou l'image porte vraiment le tatouage, le passe-haut de l'image est
correle (positivement) avec d ; dans un trou ou hors de la zone, la correlation est nulle en moyenne.
On enregistre des sommes par blocs de 8 px (num = sum n*d, nn = sum n*n, dd = sum d*d) pour 2 filtres
passe-haut : on peut ensuite recalculer la correlation a n'importe quelle taille de bloc sur CPU.

  python sig_corr.py pixelseal,stegastamp   -> v60/corr_<methode>.npz   (GPU, quelques minutes)
N'ecrit que dans v60/. Lit le dossier de la tache en lecture seule."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.dont_write_bytecode = True
T = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
V = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
from wmloc.watermarks import load_watermark  # noqa: E402

SIGMAS = [1.5, 3.0]
K_FAUX = {"pixelseal": 2, "stegastamp": 2, "trustmark": 1}


def flou(x, sigma):
    """Flou gaussien separable (x : 1 x C x H x W)."""
    r = int(3 * sigma + 0.5); t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum(); c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def sommes(img, d, dev):
    x = torch.from_numpy(img.astype(np.float32)).to(dev).permute(2, 0, 1)[None]
    d = torch.from_numpy(d.astype(np.float32)).to(dev).permute(2, 0, 1)[None]
    out = []
    for s in SIGMAS:
        n = x - flou(x, s); e = d - flou(d, s)
        st = torch.stack([(n * e).sum(1), (n * n).sum(1), (e * e).sum(1)], 1)       # 1 x 3 x 512 x 512
        out.append(F.avg_pool2d(st, 8)[0] * 64)                                       # 3 x 64 x 64
    return torch.stack(out).cpu().numpy()


def corr128(S, bloc):
    """S : 3 x 64 x 64 (sommes par blocs de 8 px) -> correlation par blocs de `bloc` px, agrandie en 128 x 128."""
    k = bloc // 8; g = 64 // k
    s = S.reshape(3, g, k, g, k).sum((2, 4))
    c = s[0] / np.sqrt(s[1] * s[2] + 1e-9)
    return np.kron(c, np.ones((128 // g, 128 // g), np.float32))


def auc(s, y):
    s = s.ravel().astype(np.float64); y = y.ravel().astype(bool)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable") + 1.0
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meths = sys.argv[1].split(",")
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    z = np.load(T / "release/train/masks_128.npz"); G = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
    for meth in meths:
        rng = np.random.default_rng(0); t0 = time.time()
        wm = load_watermark(meth, device=dev)
        noms, splits, S, fulls = [], [], [], []
        for split in ("train", "test"):
            meta = [m for m in json.loads((T / "release" / split / "metadata.json").read_text()) if m["method"] == meth]
            if nmax:
                meta = meta[:nmax]
            for j, m in enumerate(meta):
                img = np.asarray(Image.open(T / "release" / split / "images" / f"{m['image_name']}.png").convert("RGB"))
                bits = np.array([int(c) for c in m["message"]], np.uint8)
                ev = wm.embed(img, bits).astype(np.float32)
                ef = np.mean([wm.embed(img, rng.integers(0, 2, len(bits)).astype(np.uint8)).astype(np.float32)
                              for _ in range(K_FAUX[meth])], axis=0)
                S.append(sommes(img, ev - ef, dev)); noms.append(m["image_name"]); splits.append(split)
                fulls.append(bool(m.get("full_watermark", False)))
                if j % 50 == 0:
                    print(f"{meth} {split} {j + 1}/{len(meta)} ({time.time() - t0:.0f} s)", flush=True)
        S = np.stack(S).astype(np.float32)
        np.savez_compressed(V / f"corr_{meth}.npz", names=np.array(noms), splits=np.array(splits), sums=S,
                            sigmas=np.array(SIGMAS))
        print(f"ecrit corr_{meth}.npz {S.shape} ({time.time() - t0:.0f} s)", flush=True)
        # diagnostic sur le train : AUC par image (trou / tatoue), par regime
        for si, sg in enumerate(SIGMAS):
            for bloc in (8, 16, 32, 64):
                for full in (True, False):
                    a = [auc(corr128(S[i, si], bloc), G[n]) for i, n in enumerate(noms)
                         if splits[i] == "train" and fulls[i] == full]
                    a = np.array([v for v in a if not np.isnan(v)])
                    print(f"  AUC {meth} sigma={sg} bloc={bloc:2d} full={full!s:5s} n={len(a)} : moy {a.mean():.3f} "
                          f"med {np.median(a):.3f}", flush=True)
        del wm
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
