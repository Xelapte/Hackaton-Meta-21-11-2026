"""Tâche 1 : post-traitement des probabilités du U-Net, mesuré sans optimisme (validation croisée imbriquée).

Entrée : <nom>_oof.npz écrit par --mode unet (probabilités hors pli des 510 images du train, et du test).
Règle « plancher » : sur une image tatouée à 1 %, prédire tout à 0 rapporte ~0,5 ; une petite zone fausse
rapporte moins. Après seuillage (seuil t), si la classe minoritaire prédite couvre moins de `a` de l'image,
ou si sa confiance moyenne |2p - 1| est sous `c`, on prédit la classe majoritaire seule.
Les paramètres (t, a, c), par méthode, sont choisis sur 4 plis et mesurés sur le 5e : le score affiché
est donc celui qu'on peut attendre sur des images nouvelles. Puis réglage sur tout le train -> test.

  python watermark/postproc.py --oof $OFFICIEL/Hackathon_Watermark_Localisation_Task/output/unet_feats_cv5_oof.npz
(léger, CPU : peut tourner sur le nœud de login)
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.baseline import METHODES, ecrire_soumission, journal  # noqa: E402

SEUILS = np.round(np.linspace(0.3, 0.7, 9), 3)
PARTS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
CONFS = [0.0, 0.2, 0.4, 0.6]


def miou_lot(B, G):
    """mIoU officiel image par image, vectorisé. B, G : N x 128 x 128 bool."""
    s = np.zeros(len(B))
    for b, g in ((B, G), (~B, ~G)):
        inter = (b & g).sum((1, 2)); union = (b | g).sum((1, 2))
        s += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return s / 2


def appliquer(P, t, a, c):
    B = P >= t
    frac = B.mean((1, 2)); maj = frac >= 0.5
    mino = np.where(maj[:, None, None], ~B, B)
    part_min = mino.mean((1, 2))
    conf = (np.abs(2 * P - 1) * mino).sum((1, 2)) / np.maximum(mino.sum((1, 2)), 1)
    remplacer = (part_min < a) | (conf < c)
    B[remplacer] = maj[remplacer, None, None]
    return B


def meilleur(P, G, grille):
    return max(grille, key=lambda q: miou_lot(appliquer(P, *q), G).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--oof", required=True)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out", default=None, help="soumission écrite (défaut : <oof>_post.pt)")
    args = ap.parse_args()
    z = np.load(args.oof)
    names = z["names"].tolist(); meths = z["methods"]; pli = z["pli"]; P = z["probs"].astype(np.float32)
    m = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(m["names"].tolist(), m["masks"]))
    G = np.stack([M[n] for n in names]).astype(bool)
    plis = sorted(set(pli.tolist()))
    grilles = {"seuil 0,5 fixe": [(0.5, 0.0, 0.0)],
               "seuil réglé": [(t, 0.0, 0.0) for t in SEUILS],
               "seuil + plancher": list(itertools.product(SEUILS, PARTS, CONFS))}
    print(f"{len(names)} images hors pli, {len(plis)} plis")
    res = {}
    for nom, grille in grilles.items():
        sc = np.zeros(len(names))
        for meth in METHODES:
            for k in plis:                          # réglage sur les autres plis, mesure sur le pli k
                tr = (meths == meth) & (pli != k); te = (meths == meth) & (pli == k)
                q = meilleur(P[tr], G[tr], grille)
                sc[te] = miou_lot(appliquer(P[te], *q), G[te])
        res[nom] = sc
        detail = "  ".join(f"{mt} {sc[meths == mt].mean():.4f}" for mt in METHODES)
        print(f"{nom:18s}: mIoU imbriqué {sc.mean():.4f}   ({detail})")
    gain = res["seuil + plancher"] - res["seuil réglé"]
    print(f"gain du plancher : {gain.mean():+.4f} (écart-type par pli "
          f"{np.std([gain[pli == k].mean() for k in plis]):.4f})")

    # réglage final sur tout le train, appliqué au test
    params = {mt: meilleur(P[meths == mt], G[meths == mt], grilles["seuil + plancher"]) for mt in METHODES}
    print("paramètres retenus (t, a, c) :", {k: tuple(float(x) for x in v) for k, v in params.items()})
    Pt = z["test_probs"].astype(np.float32); tm = z["test_methods"]; tn = z["test_names"].tolist()
    Bt = np.zeros(Pt.shape, bool)
    for mt in METHODES:
        Bt[tm == mt] = appliquer(Pt[tm == mt], *params[mt])
    noms_test = [x["image_name"] for x in json.loads((args.data / "test" / "metadata.json").read_text())]
    assert set(noms_test) == set(tn)
    out = args.out or str(args.oof).replace("_oof.npz", "_post.pt")
    ecrire_soumission([{"image_name": n} for n in tn], list(Bt.astype(np.uint8)), out)
    journal({"mode": "postproc_plancher", "oof": Path(args.oof).name,
             "params": {k: [float(x) for x in v] for k, v in params.items()}}, float(res["seuil + plancher"].mean()))


if __name__ == "__main__":
    main()
