"""v60 / etape 1 : reproduire en local le meilleur existant (H3b de l'equipe) sur les plis geles.
CPU leger (numpy). Ne fait que LIRE les artefacts de Hackathon_Watermark_Localisation_Task/output et depot-coequipier.
  H3b carte  = soumission n 1412 (public 0,5042)
  H3b unet   = meme regle avec les probabilites hors pli du U-Net comme repli (mesure 0,5208, jamais soumis)
On ajoute le detail par regime (full_watermark) et une analyse des boites enregistrees (20 par image)."""
import argparse, json, sys
from pathlib import Path
import numpy as np

T = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
W = Path("/p/scratch/training2666/huang25/Epi84/depot-coequipier")
sys.dont_write_bytecode = True
sys.path.insert(0, str(W))
from watermark.protocole import K, METHODES, charger_plis, gain_robuste, miou_lot, resume, texte  # noqa
from watermark import regle  # noqa


def ns(repli):
    return argparse.Namespace(data=T / "release", out_dir=str(T / "output"), cartes=str(T / "output/cartes_feats.npz"),
                              repli=repli, oof=str(T / "output/unet_h5a_feats_oof.npz"), fin=False)


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool); u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def main():
    meta = {m["image_name"]: m for m in json.loads((T / "release/train/metadata.json").read_text())}
    out = {}
    for repli in ("carte", "unet"):
        D, _ = regle.charger(ns(repli), "train")
        sc, choix = regle.evaluer_imbrique(D, menus=False)
        full = np.array([bool(meta[n]["full_watermark"]) for n in D.noms])
        print(f"H3b repli {repli:5s}: {texte(resume(sc, D.pi, D.meths))}", flush=True)
        for m in METHODES:
            for f in (True, False):
                s = (D.meths == m) & (full == f)
                print(f"     {m:10s} full={f!s:5s} n={s.sum():3d}  mIoU {sc[s].mean():.4f}   (tout a {int(f)} : "
                      f"{miou_lot(np.full(D.G[s].shape, f), D.G[s]).mean():.4f})")
        out[repli] = sc
        np.save(Path(__file__).parent / f"logs/base_scores_{repli}.npy", sc)
    print("gain unet/carte :", gain_robuste(out["carte"], out["unet"], D.pi))
    # --- analyse des boites (images partielles) : la boite n 1 est-elle la bonne ? egalites de score ?
    print("\nBoites (images partielles, train) : IoU128 de la boite 1, de la boite 'consensus', oracle des 20")
    for m in METHODES:
        R = {}
        for f in sorted((T / "output/rect").glob(f"train_{m}*.npz")):
            z = np.load(f)
            for n, b, s in zip(z["names"].tolist(), z["boites"], z["full"]):
                R[n] = (b, float(s))
        i1, ic, io, nt, sb = [], [], [], [], []
        for j, n in enumerate(D.noms):
            if D.meths[j] != m or meta[n]["full_watermark"]:
                continue
            b, sfull = R[n]; g = D.G[j]
            top = b[b[:, 4] >= b[0, 4] - 1e-6]                      # boites a egalite avec la meilleure
            cons = np.median(top[:, :4], axis=0)
            i1.append(iou(regle.masque_boite(b[0]), g)); ic.append(iou(regle.masque_boite(cons), g))
            io.append(max(iou(regle.masque_boite(bb), g) for bb in b)); nt.append(len(top)); sb.append(b[0, 4])
        i1, ic, io, nt, sb = map(np.array, (i1, ic, io, nt, sb))
        print(f"  {m:10s} n={len(i1)}  boite1 {i1.mean():.3f} (med {np.median(i1):.3f})  consensus {ic.mean():.3f}  "
              f"oracle20 {io.mean():.3f}  nb egalites moyen {nt.mean():.1f}  score boite1 med {np.median(sb):.2f}")
        for lo in (0.6, 0.7, 0.8, 0.9, 0.99):
            s = sb >= lo
            if s.any():
                print(f"       score>={lo:.2f} : n={s.sum():3d}  boite1 {i1[s].mean():.3f}  consensus {ic[s].mean():.3f}  oracle20 {io[s].mean():.3f}")


if __name__ == "__main__":
    main()
