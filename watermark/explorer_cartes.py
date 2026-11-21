"""Tâche 1 : exploration. Quelle carte par position localise le mieux le tatouage ?

Pour chaque image du train, on calcule des cartes directes (sans fenêtres glissantes) :
  - PixelSeal : canal de détection (canal 0 des logits par position) et corrélation au message
    (moyenne de logit × signe du bit connu), à plusieurs résolutions d'entrée ;
  - TrustMark : corrélation au message des logits par position (layer4), à plusieurs résolutions.
Chaque carte est remise sur la grille 128 × 128 ; on donne le mIoU officiel au meilleur seuil global,
à côté des références « tout à 0 », « tout à 1 » et de la carte du mode window si --maps est donné.
Sortie : output/cartes_explo.npz (cartes 128 × 128 par image, pour réutilisation).

  bash slurm/soumettre.sh 2 wm_explo --venv watermark python watermark/explorer_cartes.py \\
      --maps $OFFICIEL/Hackathon_Watermark_Localisation_Task/output/window_maps.npz
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.baseline import score  # noqa: E402


def vers128(m):
    """Carte h × w (tenseur) -> 128 × 128 (numpy), interpolation bilinéaire."""
    return F.interpolate(m[None, None].float(), size=128, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def meilleur_seuil(cartes, gts, n=41):
    v = np.concatenate([c.ravel() for c in cartes])
    seuils = np.quantile(v, np.linspace(0.02, 0.98, n))
    return max((score([(c >= s).astype(np.uint8) for c in cartes], gts), float(s)) for s in seuils)


@torch.no_grad()
def cartes_pixelseal(wm, x, signe, res):
    """x : 1 × 3 × 512 × 512 dans [0, 1]. Renvoie (détection, corrélation) à la résolution d'entrée res."""
    xr = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True) if res != 512 else x
    d = wm.det
    lat = d.convnext(d.preprocess(xr))
    emb = d.pixel_decoder.output_upscaling(lat)
    lin = d.pixel_decoder.linear
    pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
    det = pp[0, 0]
    corr = (pp[0, 1:] * signe[:, None, None]).mean(0)
    return det, corr


@torch.no_grad()
def carte_trustmark(wm, x, signe, res):
    n = wm.net
    xr = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    h = n.maxpool(n.relu(n.bn1(n.conv1(xr))))
    for k in ("layer1", "layer2", "layer3", "layer4"):
        h = getattr(n, k)(h)
    pp = torch.einsum("bchw,oc->bohw", h, n.fc.weight) + n.fc.bias[None, :, None, None]
    return (pp[0] * signe[:, None, None]).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--tache", type=Path, default=off)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out", default=str(off / "output" / "cartes_explo.npz"))
    ap.add_argument("--maps", default=None, help="window_maps.npz, pour comparer")
    ap.add_argument("--res_ps", default="256,512,768", help="résolutions d'entrée PixelSeal")
    ap.add_argument("--res_tm", default="245,512,768", help="résolutions d'entrée TrustMark")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(args.tache))
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark

    ds = LocalisationDataset(args.data, "train")
    items = [ds[i] for i in range(len(ds))]
    print(f"{len(items)} images de train")
    fenetres = None
    if args.maps:
        z = np.load(args.maps)
        fenetres = dict(zip(z["names"].tolist(), z["maps"].astype(np.float32)))

    sauve = {}
    for meth in ["pixelseal", "trustmark", "stegastamp"]:
        its = [it for it in items if it["method"] == meth]
        gts = [it["mask128"] for it in its]
        frac = np.mean([g.mean() for g in gts])
        print(f"\n=== {meth} ({len(its)} images, part tatouée moyenne {frac:.3f}, "
              f"full_watermark {np.mean([bool(it.get('full_watermark')) for it in its]):.2f})")
        print(f"  tout à 0 : {score([np.zeros_like(g) for g in gts], gts):.4f}   "
              f"tout à 1 : {score([np.ones_like(g) for g in gts], gts):.4f}")
        if fenetres is not None:
            sc, s = meilleur_seuil([fenetres[it["image_name"]] for it in its], gts)
            print(f"  window (baseline)       : {sc:.4f} (seuil {s:.3f})")
        if meth == "stegastamp":
            continue
        wm = load_watermark(meth, models_dir=args.tache / "models", device=dev)
        cartes = {}
        t0 = time.time()
        for it in its:
            x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(dev)
            signe = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).to(dev)
            if meth == "pixelseal":
                for r in map(int, args.res_ps.split(",")):
                    det, corr = cartes_pixelseal(wm, x, signe, r)
                    cartes.setdefault(f"ps_det_{r}", []).append(vers128(det))
                    cartes.setdefault(f"ps_corr_{r}", []).append(vers128(corr))
            else:
                for r in map(int, args.res_tm.split(",")):
                    cartes.setdefault(f"tm_corr_{r}", []).append(vers128(carte_trustmark(wm, x, signe, r)))
        print(f"  ({time.time() - t0:.0f} s)")
        for k, cs in cartes.items():
            sc, s = meilleur_seuil(cs, gts)
            sc_neg, _ = meilleur_seuil([-c for c in cs], gts)       # au cas où le signe serait inversé
            print(f"  {k:22s}: {sc:.4f} (seuil {s:+.3f})   signe inversé : {sc_neg:.4f}")
            for it, c in zip(its, cs):
                sauve[f"{k}/{it['image_name']}"] = c.astype(np.float16)
        del wm; torch.cuda.empty_cache()
    np.savez_compressed(args.out, **sauve)
    print(f"\nCartes enregistrées : {args.out}")


if __name__ == "__main__":
    main()
