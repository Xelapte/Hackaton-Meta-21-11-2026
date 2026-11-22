#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v62 - Decision finale a partir des probabilites des modeles de segmentation (v61 / v62). Numpy seulement (CPU).

Pour chaque methode, on moyenne les probabilites de plusieurs passes (memes plis de CV dans toutes les passes : pli = rang du nom
% 5), puis on compare, en validation IMBRIQUEE (reglage sur 4 plis, score sur le 5e) :
  - seuil : masque = proba >= t (t regle dans les plis) ;
  - emiou : par image, on choisit le nombre k de cases a 1 qui maximise le mIoU ATTENDU (idee de v60/decide.py : la metrique
    moyenne l'IoU des deux classes, donc la petite classe pese autant que la grande), apres recalibrage p' = sig(a logit p + b).
La variante retenue par methode est celle qui gagne en imbrique ; ses parametres sont ensuite regles sur tout le train pour le test.
Usage : python decide62.py --runs pixelseal=v61/r2,v62/L5 trustmark=v61/r2 stegastamp=v61/r1 [--write subs/x.pt]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

B = Path("/p/scratch/training2666/huang25/Epi84/unlearning")
TASK = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
METHODS = ["pixelseal", "trustmark", "stegastamp"]
TS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
A_ = [0.5, 0.7, 1.0, 1.4, 2.0]
B_ = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]


def miou_batch(pred, gt):
    p, g = pred.reshape(len(pred), -1).astype(bool), gt.reshape(len(gt), -1).astype(bool)
    out = np.zeros(len(p))
    for c in (True, False):
        pc, gc = (p == c), (g == c)
        u, i = (pc | gc).sum(1), (pc & gc).sum(1)
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


def emiou(P, a, b):
    """Masque qui maximise le mIoU attendu, image par image (P : n x 128 x 128 probabilites)."""
    n = len(P); f = np.clip(P.reshape(n, -1).astype(np.float64), 1e-4, 1 - 1e-4)
    q = 1 / (1 + np.exp(-(a * np.log(f / (1 - f)) + b)))
    o = np.argsort(-q, axis=1); sp = np.take_along_axis(q, o, 1); N = q.shape[1]
    S = np.concatenate([np.zeros((n, 1)), np.cumsum(sp, 1)], 1); SN = S[:, -1:]; k = np.arange(N + 1)[None]
    i1 = S / np.maximum(k + SN - S, 1e-9)
    i0 = ((N - k) - (SN - S)) / np.maximum((N - k) + (k - S), 1e-9)
    i1[:, 0] = 0.0; i0[:, N] = 0.0            # chaque image contient du watermark et du non-watermark (ou presque)
    kb = np.argmax(i1 + i0, axis=1); kb = np.maximum(kb, 1)
    r = np.empty_like(o); np.put_along_axis(r, o, np.broadcast_to(np.arange(N), o.shape), 1)
    return (r < kb[:, None]).reshape(P.shape)


def charger(spec):
    """'v61/r2,v62/L5' -> moyenne des probabilites (train_oof, test_cv) des passes listees."""
    oof, tcv, tr_names, te_names, folds = [], [], None, None, None
    for s in spec.split(","):
        d, tag = s.split("/")
        flip = tag.endswith("+f")           # 'v63/L9+f' : moyenne avec la prediction TTA miroir
        tag = tag[:-2] if flip else tag
        z = np.load(B / d / f"probs_{M}_{tag}.npz")
        assert "train_oof" in z.files, f"{d}/{tag} : pas de train_oof (run inacheve ?)"
        n1, n2 = [str(x) for x in z["train_names"]], [str(x) for x in z["test_names"]]
        if tr_names is None:
            tr_names, te_names, folds = n1, n2, z["folds"]
        assert n1 == tr_names and n2 == te_names and (z["folds"] == folds).all(), "passes incompatibles"
        o, t = z["train_oof"].astype(np.float32), z["test_cv"].astype(np.float32)
        if flip:
            o, t = (o + z["train_oof_flip"].astype(np.float32)) / 2, (t + z["test_cv_flip"].astype(np.float32)) / 2
        oof.append(o); tcv.append(t)
    return np.mean(oof, 0), np.mean(tcv, 0), tr_names, te_names, folds


def regler(P, G, var):
    if var == "seuil":
        return {"t": max(TS, key=lambda t: miou_batch(P >= t, G).mean())}
    a, b = max(((a, b) for a in A_ for b in B_), key=lambda q: miou_batch(emiou(P, *q), G).mean())
    return {"a": a, "b": b}


def appliquer(P, p):
    return P >= p["t"] if "t" in p else emiou(P, p["a"], p["b"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--write", default=None)
    ap.add_argument("--force", default=None, help="seuil | emiou : impose la variante pour toutes les methodes")
    a = ap.parse_args()
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    te_meta = json.loads((TASK / "release/test/metadata.json").read_text())
    tot, test_masks, resume = [], {}, {}
    for spec in a.runs:
        M, runs = spec.split("=")
        P, T, ntr, nte, folds = charger(runs)
        G = np.stack([gt[n] for n in ntr]).astype(bool)
        res = {}
        for var in ("seuil", "emiou"):
            sc = np.zeros(len(P))
            for k in range(5):
                tr, te = folds != k, folds == k
                sc[te] = miou_batch(appliquer(P[te], regler(P[tr], G[tr], var)), G[te])
            res[var] = sc
        brut = miou_batch(P >= 0.5, G).mean()
        var = a.force or max(res, key=lambda v: res[v].mean())
        p = regler(P, G, var)
        print(f"[{M} {runs}] seuil 0.5 : {brut:.4f} | imbrique seuil : {res['seuil'].mean():.4f} | imbrique emiou : "
              f"{res['emiou'].mean():.4f} -> {var} {p}", flush=True)
        tot.append(res[var]); resume[M] = {"runs": runs, "variante": var, "params": p,
                                           "imbrique": {v: float(s.mean()) for v, s in res.items()}}
        for n, m in zip(nte, appliquer(T, p)):
            test_masks[n] = m
    print(f"mIoU imbrique global : {np.concatenate(tot).mean():.4f}")
    if a.write:
        import torch
        names = [m["image_name"] for m in te_meta]
        assert all(n in test_masks for n in names)
        out = B / "v62" / a.write
        assert not out.exists(), f"{out} existe deja"
        masks = torch.from_numpy(np.stack([test_masks[n] for n in names])).bool()
        assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
        torch.save({"image_names": names, "masks": masks}, out)
        chk = torch.load(out, weights_only=True); assert chk["image_names"] == names
        assert os.path.getsize(out) < 20e6
        Path(str(out)[:-3] + ".json").write_text(json.dumps({"choix": resume, "imbrique_global": float(np.concatenate(tot).mean())}, indent=1))
        print(f"ecrit {out}, part de 1 : {masks.float().mean():.3f}")
