"""Tâche 1 : exploration. « Re-tatouer et corréler » localise-t-il le tatouage ?

Idée : les embedders sont fournis et le message de chaque image est connu. R' = embed(I, message) - I
approche le résidu ajouté à l'origine. Là où l'image est tatouée, ses hautes fréquences corrèlent
localement avec R' ; ailleurs non. On calcule une corrélation normalisée par blocs (--bloc pixels),
remise sur la grille 128 × 128, et on donne le mIoU au meilleur seuil, par méthode et selon full_watermark.
Sortie : output/reembed_<méthode>.npz (cartes 128 × 128, train et test si --test).

  bash slurm/soumettre.sh 2 wm_reembed --venv watermark python watermark/explorer_reembed.py
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
from watermark.baseline import carte_reembed, score, seuil_optimal  # noqa: E402


def carte_corr(img_u8, re_u8, bloc, k, dev):
    return carte_reembed(img_u8, re_u8, bloc, dev, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--tache", type=Path, default=off)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--methodes", default="stegastamp,trustmark,pixelseal")
    ap.add_argument("--blocs", default="16,32,64", help="tailles de bloc de la corrélation (pixels)")
    ap.add_argument("--k", type=int, default=5, help="noyau du passe-haut")
    ap.add_argument("--test", action="store_true", help="calculer aussi les cartes du test")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(args.tache))
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark

    splits = ["train"] + (["test"] if args.test else [])
    data = {s: LocalisationDataset(args.data, s) for s in splits}
    blocs = [int(b) for b in args.blocs.split(",")]
    for meth in args.methodes.split(","):
        wm = load_watermark(meth, models_dir=args.tache / "models", device=dev)
        t0 = time.time(); cartes = {b: {} for b in blocs}; tr = []
        for s in splits:
            ds = data[s]
            for i, m in enumerate(ds.meta):
                if m["method"] != meth:
                    continue
                it = ds[i]
                re = wm.embed(it["image"], it["message"])
                for b in blocs:
                    cartes[b][it["image_name"]] = carte_corr(it["image"], re, b, args.k, dev)
                if s == "train":
                    tr.append(it)
        print(f"\n=== {meth} : {len(tr)} train ({time.time() - t0:.0f} s)")
        gts = [it["mask128"] for it in tr]
        full = np.array([bool(it.get("full_watermark")) for it in tr])
        for b in blocs:
            cs = [cartes[b][it["image_name"]] for it in tr]
            sc, s = seuil_optimal(cs, gts)
            preds = [(c >= s).astype(np.uint8) for c in cs]
            sf = score([p for p, f in zip(preds, full) if f], [g for g, f in zip(gts, full) if f])
            sp = score([p for p, f in zip(preds, full) if not f], [g for g, f in zip(gts, full) if not f])
            print(f"  bloc {b:3d} : mIoU {sc:.4f} (seuil {s:+.4f})   full_watermark {sf:.4f}   partiel {sp:.4f}")
        np.savez_compressed(Path(args.out_dir) / f"reembed_{meth}.npz",
                            **{f"{b}/{n}": c.astype(np.float16) for b in blocs for n, c in cartes[b].items()})
        del wm; torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
