"""v60 / cartes « logits par position » MULTI-ECHELLES (PixelSeal et TrustMark), train + test.

Rappel (l'equipe) : le decodeur est un reseau convolutif ; avant la moyenne finale il donne des logits de bits
en chaque position. On les compare au message CONNU -> une carte de « presence du tatouage » (ps_512, tm_245).
Limite : le tatouage est fabrique a une taille fixe (256 px) puis etire sur la zone tatouee. Une petite zone
(ex. 100 px) contient donc un tatouage 2,5 fois trop petit pour le decodeur : les cartes existantes (image a
256-512 px) ne peuvent pas la voir. Ici on AGRANDIT l'image (x1,5 a x4, et aussi dans une seule direction pour
les zones allongees) avant de passer le decodeur : chaque echelle « voit » les zones d'une taille donnee, a
toutes les positions d'un coup (recherche dense, alors que la recherche de rectangles saute des positions).
Deux statistiques par position et par echelle (test d'hypothese de l'indice 1 du keynote) :
  c = moyenne des logits signes par le message ; a = part de bits en accord (0,5 = hasard, 1 = tatoue).
Sortie : v60/ms_<methode>.npz : maps (N, n_echelles, 2, 64, 64) float16 + diagnostic AUC sur le train.
"""
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

ECHELLES = {
    "pixelseal": [(768, 768), (1024, 1024), (1536, 1536), (2048, 2048), (512, 1024), (1024, 512),
                  (512, 2048), (2048, 512), (1024, 2048), (2048, 1024)],
    "trustmark": [(368, 368), (490, 490), (735, 735), (980, 980), (1470, 1470), (1960, 1960),
                  (490, 980), (980, 490), (980, 1960), (1960, 980)],
}


def par_position(wm, meth, x):
    """x : 1 x 3 x H x W dans [0, 1]. Renvoie les logits par position (nbits x h x w), h, w <= 64."""
    if meth == "pixelseal":
        d = wm.det
        emb = d.pixel_decoder.output_upscaling(d.convnext(d.preprocess(x)))
        if max(emb.shape[-2:]) > 64:
            emb = F.adaptive_avg_pool2d(emb, (min(64, emb.shape[-2]), min(64, emb.shape[-1])))
        lin = d.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb.float(), lin.weight.float()) + lin.bias.float()[None, :, None, None]
        return pp[0, 1:]
    n = wm.net; h = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
    for k in ("layer1", "layer2", "layer3", "layer4"):
        h = getattr(n, k)(h)
    if max(h.shape[-2:]) > 64:
        h = F.adaptive_avg_pool2d(h, (min(64, h.shape[-2]), min(64, h.shape[-1])))
    pp = torch.einsum("bchw,oc->bohw", h.float(), n.fc.weight.float()) + n.fc.bias.float()[None, :, None, None]
    return pp[0]


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
        t0 = time.time(); wm = load_watermark(meth, device=dev); ech = ECHELLES[meth]
        amp = dev == "cuda"; rate = set()
        noms, splits, fulls, M = [], [], [], []
        for split in ("train", "test"):
            meta = [m for m in json.loads((T / "release" / split / "metadata.json").read_text()) if m["method"] == meth]
            if nmax:
                meta = meta[:nmax]
            for j, m in enumerate(meta):
                img = np.asarray(Image.open(T / "release" / split / "images" / f"{m['image_name']}.png").convert("RGB"))
                x0 = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(dev)
                signe = torch.tensor([int(c) * 2 - 1 for c in m["message"]], dtype=torch.float32, device=dev)
                out = np.zeros((len(ech), 2, 64, 64), np.float32)
                for e, (H, W) in enumerate(ech):
                    if e in rate:
                        continue
                    try:
                        with torch.no_grad():
                            x = F.interpolate(x0, size=(H, W), mode="bilinear", align_corners=False, antialias=True)
                            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                                pp = par_position(wm, meth, x)
                            s = pp.float() * signe[:, None, None]
                            st = torch.stack([s.mean(0), (s > 0).float().mean(0)])[None]
                            if not torch.isfinite(st).all():
                                raise FloatingPointError("valeurs non finies")
                            out[e] = F.interpolate(st, size=(64, 64), mode="bilinear", align_corners=False)[0].cpu().numpy()
                    except FloatingPointError:
                        if amp:                      # demi-precision instable : on repasse en float32 pour toute la suite
                            amp = False; print(f"{meth} : demi-precision abandonnee (echelle {H}x{W})", flush=True)
                        else:
                            rate.add(e)
                    except RuntimeError as ex:        # memoire insuffisante : on abandonne cette echelle
                        rate.add(e); print(f"{meth} : echelle {H}x{W} abandonnee ({str(ex)[:80]})", flush=True)
                        if dev == "cuda":
                            torch.cuda.empty_cache()
                M.append(out); noms.append(m["image_name"]); splits.append(split); fulls.append(bool(m.get("full_watermark", False)))
                if j % 50 == 0:
                    print(f"{meth} {split} {j + 1}/{len(meta)} ({time.time() - t0:.0f} s)", flush=True)
        M = np.stack(M)
        np.savez_compressed(V / f"ms_{meth}.npz", names=np.array(noms), splits=np.array(splits),
                            maps=M.astype(np.float16), echelles=np.array(ech), rate=np.array(sorted(rate)))
        print(f"ecrit ms_{meth}.npz {M.shape} ({time.time() - t0:.0f} s) ; echelles abandonnees : {sorted(rate)}", flush=True)
        for e, (H, W) in enumerate(ech):
            for si, sn in enumerate(("c", "a")):
                txt = []
                for full in (True, False):
                    a = [auc(M[i, e, si], G[n][::2, ::2]) for i, n in enumerate(noms) if splits[i] == "train" and fulls[i] == full]
                    a = np.array([v for v in a if not np.isnan(v)])
                    txt.append(f"full={full!s:5s} AUC moy {a.mean():.3f} med {np.median(a):.3f}")
                tr = [i for i in range(len(noms)) if splits[i] == "train"]
                mx = np.array([M[i, e, 1].max() for i in tr]); fl = np.array([fulls[i] for i in tr])
                print(f"  {meth} {H:4d}x{W:4d} {sn} : {txt[0]} | {txt[1]} | max(a) moyen full {mx[fl].mean():.3f} partiel {mx[~fl].mean():.3f}", flush=True)
        del wm
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
