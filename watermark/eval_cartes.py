"""Tâche 1 : évaluer des cartes déjà calculées selon le protocole (plis gelés, réglages dans les plis). CPU, léger.

Pour chaque canal de cartes_feats.npz (mode cartes de baseline.py) et pour window_maps.npz :
  - AUC par pixel (tatoué contre propre), par méthode ;
  - mIoU avec un seuil par méthode réglé sur les 4 plis d'entraînement, mesuré sur le 5e.
Puis H2 : classer chaque image « surtout tatouée » / « surtout propre » avec la moyenne de sa carte
(seuil réglé dans les plis) et prédire la classe seule : référence « classe majoritaire devinée ».

  python watermark/eval_cartes.py --cartes .../cartes_feats.npz --window .../window_maps.npz
  python watermark/eval_cartes.py --fake
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.protocole import K, METHODES, charger_plis, miou_lot, resume, texte  # noqa: E402

CANAL = {"pixelseal": "ps_", "trustmark": "tm_", "stegastamp": "ss_"}


def auc(scores, labels):
    """AUC (Mann-Whitney) avec sous-échantillonnage pour la vitesse."""
    s = np.asarray(scores, np.float64).ravel(); y = np.asarray(labels).ravel().astype(bool)
    if len(s) > 400000:
        i = np.random.default_rng(0).choice(len(s), 400000, replace=False); s, y = s[i], y[i]
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def seuil_plis(C, G, meths, pi, quantiles=np.linspace(0.02, 0.98, 49)):
    """mIoU par image avec un seuil par méthode choisi sur les plis d'entraînement."""
    sc = np.zeros(len(C))
    for m in METHODES:
        for k in range(K):
            tr = (meths == m) & (pi != k); te = (meths == m) & (pi == k)
            if not te.any():
                continue
            cand = np.quantile(C[tr], quantiles)
            best = max(cand, key=lambda t: miou_lot(C[tr] >= t, G[tr]).mean())
            sc[te] = miou_lot(C[te] >= best, G[te])
    return sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--cartes", default=str(off / "output" / "cartes_feats.npz"))
    ap.add_argument("--window", default=str(off / "output" / "window_maps.npz"))
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    if args.fake:
        rng = np.random.default_rng(0); n = 60
        noms = [f"img{i}" for i in range(n)]; meths = np.array([METHODES[i % 3] for i in range(n)])
        G = np.zeros((n, 128, 128), bool)
        for i in range(n):
            y, x = rng.integers(0, 64, 2); G[i, y:y + 40, x:x + 40] = True
        cartes = {"ps_512": G + rng.normal(0, 1, G.shape), "tm_224": G + rng.normal(0, 1, G.shape),
                  "ss_16": G + rng.normal(0, 2, G.shape)}
        pi = np.arange(n) % K
    else:
        meta = json.loads((args.data / "train" / "metadata.json").read_text())
        noms = [m["image_name"] for m in meta]; meths = np.array([m["method"] for m in meta])
        z = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
        G = np.stack([M[x] for x in noms]).astype(bool)
        plis = charger_plis(); pi = np.array([plis[x] for x in noms])
        zc = np.load(args.cartes); F = dict(zip(zc["names"].tolist(), zc["feats"])); can = zc["canaux"].tolist()
        cartes = {c: np.stack([F[x][j] for x in noms]).astype(np.float32) for j, c in enumerate(can)}
        if args.window and Path(args.window).exists():
            zw = np.load(args.window); W = dict(zip(zw["names"].tolist(), zw["maps"]))
            cartes["window"] = np.stack([W[x] for x in noms]).astype(np.float32)
    print("H1 : carte seule, seuil par méthode réglé dans les plis", flush=True)
    meilleure = {}
    for c, C in cartes.items():
        for m in METHODES:
            if c != "window" and not c.startswith(CANAL[m]):
                continue
            sel = meths == m
            sc = seuil_plis(C[sel], G[sel], meths[sel], pi[sel])
            r = resume(sc, pi[sel], meths[sel])
            a = auc(C[sel], G[sel])
            print(f"  {m:10s} {c:8s}: AUC pixel {a:.3f}   mIoU {r['moyenne']:.4f} ± {r['ecart_type']:.4f}", flush=True)
            if m not in meilleure or r["moyenne"] > meilleure[m][1]:
                meilleure[m] = (c, r["moyenne"])
    print("  meilleure carte par méthode :", {m: f"{c} {v:.4f}" for m, (c, v) in meilleure.items()}, flush=True)

    print("\nH2 : classe majoritaire devinée par la moyenne de la carte (seuil réglé dans les plis)", flush=True)
    z0 = miou_lot(np.zeros_like(G), G); u1 = miou_lot(np.ones_like(G), G)
    frac = G.reshape(len(G), -1).mean(1)
    sc = np.zeros(len(G)); prec = []
    for m in METHODES:
        c = meilleure[m][0]; moy = cartes[c].reshape(len(G), -1).mean(1)
        for k in range(K):
            tr = (meths == m) & (pi != k); te = (meths == m) & (pi == k)
            t = max(np.unique(moy[tr]), key=lambda t: ((moy[tr] >= t) == (frac[tr] >= .5)).mean())
            maj = moy[te] >= t
            sc[te] = np.where(maj, u1[te], z0[te]); prec.append(float((maj == (frac[te] >= .5)).mean()))
        print(f"  {m:10s} ({c}) : précision {np.mean(prec[-K:]):.3f}", flush=True)
    print(f"  classe majoritaire devinée : {texte(resume(sc, pi, meths))}", flush=True)


if __name__ == "__main__":
    main()
