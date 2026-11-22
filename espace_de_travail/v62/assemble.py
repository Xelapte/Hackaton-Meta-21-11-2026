#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v61 - Evaluation honnete (vraies images d'entrainement) et construction de la soumission. Numpy seulement (pas de GPU).

Pour chaque methode on lit probs_<methode>_<tag>.npz (ecrit par seg_synth*.py) :
  train_synth : probabilites du modele entraine UNIQUEMENT sur le synthetique (les vraies images train n'ont jamais ete vues)
  train_oof   : probabilites hors-pli apres affinage en validation croisee 5 plis sur les vraies images
  test_synth / test_cv : idem sur le test (test_cv = moyenne des 5 modeles)
On choisit par methode la variante et le seuil qui maximisent le mIoU officiel sur le train, puis on ecrit la soumission.
Usage : python assemble.py --tags pixelseal=r2 trustmark=r2 stegastamp=r2 [--write subs/v61_xxx.pt]
"""
import argparse, json, os
from pathlib import Path

import numpy as np

TASK = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
OUT = Path("/p/scratch/training2666/huang25/Epi84/unlearning/v62")
METHODS = ["pixelseal", "trustmark", "stegastamp"]
TS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def miou_batch(pred, gt):
    """Score officiel (wmloc.utils.miou_image) vectorise : moyenne sur les classes 0/1 de |P&G| / |P|G| (1 si union vide)."""
    p, g = pred.reshape(len(pred), -1).astype(bool), gt.reshape(len(gt), -1).astype(bool)
    out = np.zeros(len(p))
    for c in (True, False):
        pc, gc = (p == c), (g == c)
        union, inter = (pc | gc).sum(1), (pc & gc).sum(1)
        out += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


def post(prob, thr, min_area=0, rect_fill=1.01):
    """Post-traitement simple d'une carte de probabilite 128 x 128 -> masque 0/1.
    1) seuil ; si rien ne depasse, on garde les cases >= moitie du maximum (un masque vide est toujours faux : chaque image
       contient du watermark) ; 2) on supprime les ilots de moins de min_area cases (sauf le plus grand) et on bouche les
       trous de moins de min_area cases ; 3) une composante presque rectangulaire (remplissage >= rect_fill de son rectangle
       englobant, et < 45 % de l'image) est remplacee par ce rectangle (la moitie des patchs reels sont des rectangles)."""
    from scipy import ndimage as ndi
    m = prob >= thr
    if not m.any():
        m = prob >= 0.5 * prob.max()
    if min_area <= 0 and rect_fill > 1:
        return m
    lab, n = ndi.label(m)
    if n:
        areas = np.asarray(ndi.sum(m, lab, index=np.arange(1, n + 1)))
        big = int(np.argmax(areas)) + 1
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if areas[i - 1] < min_area and i != big:
                m[lab == i] = False
            elif rect_fill <= 1 and areas[i - 1] < 0.45 * m.size:
                h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                if areas[i - 1] / float(h * w) >= rect_fill:
                    m[sl] = True
    if min_area > 0:
        lab0, n0 = ndi.label(~m)
        if n0 > 1:
            a0 = np.asarray(ndi.sum(~m, lab0, index=np.arange(1, n0 + 1)))
            big0 = int(np.argmax(a0)) + 1
            for i in range(1, n0 + 1):
                if a0[i - 1] < min_area and i != big0:
                    m[lab0 == i] = True
    return m


POSTS = [(0, 1.01), (8, 1.01), (16, 1.01), (32, 1.01), (16, 0.9), (16, 0.8), (16, 0.7), (32, 0.8), (32, 0.7), (64, 0.8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="methode=tag[:variante[:seuil]] ; variante = cv | synth | auto")
    ap.add_argument("--write", default=None)
    ap.add_argument("--post", default="none", help="none | auto (choisit min_area/rect_fill sur le train) | min_area,rect_fill")
    a = ap.parse_args()
    tr = json.loads((TASK / "release/train/metadata.json").read_text())
    te = json.loads((TASK / "release/test/metadata.json").read_text())
    z = np.load(TASK / "release/train/masks_128.npz")
    gt = dict(zip(z["names"].tolist(), z["masks"]))
    full = {m["image_name"]: bool(m.get("full_watermark", False)) for m in tr}
    meth_tr = {m["image_name"]: m["method"] for m in tr}

    # reference : U-Net du coequipier (hors-pli, memes images), relu tel quel
    ref = {}
    for f in ("unet_feats_cv5_oof.npz", "unet_h5a_feats_oof.npz"):
        try:
            u = np.load(TASK / "output" / f, allow_pickle=True)
            names = [str(x) for x in u["names"]]
            sc = miou_batch(u["probs"].astype(np.float32) >= 0.5, np.stack([gt[n] for n in names]))
            ref[f] = {m: float(np.mean([s for s, n in zip(sc, names) if meth_tr[n] == m])) for m in METHODS}
            ref[f]["tout"] = float(sc.mean())
        except Exception as e:  # noqa: BLE001
            print("reference illisible", f, repr(e)[:100])
    for f, r in ref.items():
        print(f"reference coequipier {f} (hors-pli, seuil 0.5) : " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))

    choix, test_masks, per_img, resume = {}, {}, {}, {}
    for spec in a.tags:
        meth, rest = spec.split("=")
        parts = rest.split(":")
        tag, var, thr = parts[0], (parts[1] if len(parts) > 1 else "auto"), (float(parts[2]) if len(parts) > 2 else None)
        p = np.load(OUT / f"probs_{meth}_{tag}.npz")
        names = [str(x) for x in p["train_names"]]
        g = np.stack([gt[n] for n in names])
        fl = np.array([full[n] for n in names])
        table = {}
        for v, key in (("synth", "train_synth"), ("cv", "train_oof")):
            if key not in p.files:
                continue
            pr = p[key].astype(np.float32)
            table[v] = {t: miou_batch(pr >= t, g) for t in TS}
            best_t = max(TS, key=lambda t: table[v][t].mean())
            print(f"[{meth} {tag}] {v:<5} : seuil 0.5 -> {table[v][0.5].mean():.4f} (entiere {table[v][0.5][fl].mean():.4f} | patch {table[v][0.5][~fl].mean():.4f})"
                  f" ; meilleur seuil {best_t} -> {table[v][best_t].mean():.4f} (entiere {table[v][best_t][fl].mean():.4f} | patch {table[v][best_t][~fl].mean():.4f})")
        if var == "auto":
            var = max(table, key=lambda v: max(table[v][t].mean() for t in TS))
        if thr is None:
            thr = max(TS, key=lambda t: table[var][t].mean())
        sc = table[var][thr]
        pp = (0, 1.01)
        if a.post != "none":
            pr = p["train_oof" if var == "cv" else "train_synth"].astype(np.float32)
            cands = POSTS if a.post == "auto" else [tuple(float(v) for v in a.post.split(","))]
            res = {}
            for t in sorted({thr, 0.5, max(0.15, round(thr - 0.1, 2)), min(0.8, round(thr + 0.1, 2))}):
                for c in cands:
                    res[(t,) + c] = miou_batch(np.stack([post(q, t, *c) for q in pr]), g)
            for kk in sorted(res, key=lambda k: -res[k].mean())[:6]:
                print(f"      post seuil={kk[0]} min_area={kk[1]} rect_fill={kk[2]} -> {res[kk].mean():.4f} (entiere {res[kk][fl].mean():.4f} | patch {res[kk][~fl].mean():.4f})")
            bk = max(res, key=lambda k: res[k].mean())
            thr, pp, sc = bk[0], bk[1:], res[bk]
        choix[meth] = {"tag": tag, "variante": var, "seuil": thr, "post_min_area": pp[0], "post_rect_fill": pp[1],
                       "miou_train": float(sc.mean()), "miou_train_seuil_0.5_sans_post": float(table[var][0.5].mean()),
                       "miou_entiere": float(sc[fl].mean()), "miou_patch": float(sc[~fl].mean()), "n": len(names)}
        for n, s in zip(names, sc):
            per_img[n] = float(s)
        tp = p["test_cv" if var == "cv" else "test_synth"].astype(np.float32)
        for n, q in zip([str(x) for x in p["test_names"]], tp):
            test_masks[n] = post(q, thr, *pp)
        resume[meth] = choix[meth]
        print(f"   -> choix {meth} : {choix[meth]}")
    if len(per_img):
        print(f"\nmIoU local global sur {len(per_img)} vraies images train (choix ci-dessus) : {np.mean(list(per_img.values())):.4f}")
        for m in METHODS:
            v = [s for n, s in per_img.items() if meth_tr[n] == m]
            if v:
                print(f"   {m:<11} n={len(v)} : {np.mean(v):.4f}")
    if a.write:
        import torch
        names = [m["image_name"] for m in te]
        manq = [n for n in names if n not in test_masks]
        assert not manq, f"{len(manq)} images de test sans prediction (methodes manquantes ?)"
        masks = torch.from_numpy(np.stack([test_masks[n] for n in names])).bool()
        assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
        out = OUT / a.write
        assert not out.exists(), f"{out} existe deja : choisir un autre nom (on n'ecrase rien)"
        torch.save({"image_names": names, "masks": masks}, out)
        chk = torch.load(out, weights_only=True)
        assert chk["masks"].shape == (1020, 128, 128) and chk["masks"].dtype == torch.bool and chk["image_names"] == names
        size = os.path.getsize(out) / 1e6
        assert size < 20, size
        (OUT / (a.write.rsplit(".", 1)[0] + ".json")).write_text(json.dumps(
            {"choix": resume, "miou_local_global": float(np.mean(list(per_img.values()))), "reference_coequipier": ref}, indent=1))
        print(f"\necrit {out} : {tuple(chk['masks'].shape)} {chk['masks'].dtype}, {size:.1f} Mo, part de cases a 1 : {chk['masks'].float().mean():.3f}")


if __name__ == "__main__":
    main()
