"""Tâche 1 : protocole d'évaluation commun (plis gelés, références triviales, critère de gain).

  python watermark/protocole.py --creer      # une seule fois : écrit watermark/splits.json (refuse s'il existe)
  python watermark/protocole.py              # références triviales, sur les plis gelés
  python watermark/protocole.py --fake       # test sans données

Les plis : 5, stratifiés par méthode x full_watermark (seule information de type de retouche disponible),
graine 0. Ne plus jamais les modifier : toutes les expériences les utilisent (charger_plis()).
Le score d'une méthode = mIoU officiel sur le pli tenu à l'écart, moyenné sur les 5 plis (± écart-type),
avec le détail par méthode. Un gain compte s'il dépasse 0,02, dépasse l'écart-type entre plis du gain,
et est positif sur au moins 4 plis sur 5 (gain_robuste()).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
CHEMIN_PLIS = RACINE / "watermark" / "splits.json"
METHODES = ["pixelseal", "trustmark", "stegastamp"]
K = 5


def miou_lot(B, G):
    """mIoU officiel image par image (identique à wmloc.utils.miou_image), vectorisé. B, G : N x 128 x 128."""
    B = np.asarray(B).astype(bool); G = np.asarray(G).astype(bool)
    s = np.zeros(len(B))
    for b, g in ((B, G), (~B, ~G)):
        inter = (b & g).sum((1, 2)); union = (b | g).sum((1, 2))
        s += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return s / 2


def creer_plis(meta, graine=0):
    rng = np.random.default_rng(graine)
    plis = {}
    for meth in METHODES:
        for full in (False, True):
            noms = sorted(m["image_name"] for m in meta if m["method"] == meth and bool(m["full_watermark"]) == full)
            noms = [noms[i] for i in rng.permutation(len(noms))]
            for i, n in enumerate(noms):
                plis[n] = i % K
    return plis


def charger_plis(chemin=CHEMIN_PLIS):
    d = json.loads(Path(chemin).read_text())
    return d["plis"]


def resume(scores, plis_img, meths):
    """scores : mIoU par image ; renvoie dict (moyenne des plis, écart-type, par pli, par méthode)."""
    scores = np.asarray(scores); plis_img = np.asarray(plis_img); meths = np.asarray(meths)
    par_pli = np.array([scores[plis_img == k].mean() for k in range(K)])
    par_meth = {m: float(scores[meths == m].mean()) for m in METHODES if (meths == m).any()}
    par_meth_pli = {m: [float(scores[(meths == m) & (plis_img == k)].mean()) for k in range(K)] for m in par_meth}
    return {"moyenne": float(par_pli.mean()), "ecart_type": float(par_pli.std()), "par_pli": par_pli.tolist(),
            "par_methode": par_meth, "par_methode_pli": par_meth_pli}


def texte(r):
    pm = "  ".join(f"{m[:5]} {v:.4f}±{np.std(r['par_methode_pli'][m]):.4f}" for m, v in r["par_methode"].items())
    return f"{r['moyenne']:.4f} ± {r['ecart_type']:.4f}   ({pm})"


def gain_robuste(scores_a, scores_b, plis_img, seuil=0.02):
    """Gain de b sur a (mIoU par image). Robuste si > seuil, > écart-type du gain entre plis, et > 0 sur >= 4 plis."""
    d = np.asarray(scores_b) - np.asarray(scores_a); plis_img = np.asarray(plis_img)
    g = np.array([d[plis_img == k].mean() for k in range(K)])
    ok = g.mean() > seuil and g.mean() > g.std() and (g > 0).sum() >= 4
    return {"gain": float(g.mean()), "ecart_type": float(g.std()), "plis_positifs": int((g > 0).sum()),
            "par_pli": g.tolist(), "robuste": bool(ok)}


def references_triviales(G, plis_img, meths):
    """Tout à 0, tout à 1, et le meilleur des deux par méthode (choisi sur les plis d'entraînement)."""
    n = len(G)
    z = miou_lot(np.zeros_like(G), G); u = miou_lot(np.ones_like(G), G)
    best = np.zeros(n)
    for m in METHODES:
        for k in range(K):
            tr = (meths == m) & (plis_img != k); te = (meths == m) & (plis_img == k)
            best[te] = z[te] if z[tr].mean() >= u[tr].mean() else u[te]
    return {"tout à 0": z, "tout à 1": u, "meilleur des deux par méthode": best}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--creer", action="store_true")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    if args.fake:
        rng = np.random.default_rng(0)
        meta = [{"image_name": f"img{i}", "method": METHODES[i % 3], "full_watermark": bool(i % 2)} for i in range(60)]
        M = {m["image_name"]: (rng.random((128, 128)) < 0.3).astype(np.uint8) for m in meta}
        plis = creer_plis(meta)
    else:
        meta = json.loads((args.data / "train" / "metadata.json").read_text())
        z = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
        if args.creer:
            if CHEMIN_PLIS.exists():
                sys.exit(f"{CHEMIN_PLIS} existe déjà : les plis sont gelés, on ne les recrée pas.")
            plis = creer_plis(meta)
            CHEMIN_PLIS.write_text(json.dumps({"graine": 0, "k": K, "stratification": "method x full_watermark",
                                               "plis": plis}, indent=0, sort_keys=True))
            print(f"Plis écrits : {CHEMIN_PLIS}", flush=True)
        plis = charger_plis()
    noms = [m["image_name"] for m in meta]
    G = np.stack([M[n] for n in noms]); meths = np.array([m["method"] for m in meta])
    pi = np.array([plis[n] for n in noms])
    for k in range(K):
        sel = pi == k
        print(f"pli {k} : {sel.sum()} images  " + " ".join(f"{m[:5]}={(sel & (meths == m)).sum()}" for m in METHODES)
              + f"  full={np.mean([meta[i]['full_watermark'] for i in np.where(sel)[0]]):.2f}", flush=True)
    print("\nRéférences triviales (mIoU sur les plis tenus à l'écart, moyenne ± écart-type entre plis) :")
    for nom, sc in references_triviales(G, pi, meths).items():
        print(f"  {nom:32s}: {texte(resume(sc, pi, meths))}", flush=True)
    # oracle « classe majoritaire » (connaît le masque : borne, pas une méthode)
    frac = G.reshape(len(G), -1).mean(1)
    orc = miou_lot(np.broadcast_to((frac >= 0.5)[:, None, None], G.shape), G)
    print(f"  {'oracle classe majoritaire (borne)':32s}: {texte(resume(orc, pi, meths))}", flush=True)


if __name__ == "__main__":
    main()
