"""v60 / regle de decision finale (CPU leger), evaluee sur les plis geles de l'equipe (reglage sur 4 plis, mesure sur le 5e).

Meme squelette que la regle H3b de l'equipe (regle.py), par methode :
  - regime « full »   : precision de bits de l'image entiere >= tf            -> masque = probabilites >= t_full
  - regime « boite »  : sinon, score de la meilleure boite >= tb              -> masque = rectangle
  - regime « vide »   : sinon                                                 -> masque = probabilites >= t_vide
Ce qui change (chaque option est mesuree separement, tout est regle DANS les plis d'entrainement) :
  --cons     : par methode, choix entre la 1re boite de la liste et la boite « consensus » (mediane des boites
               a egalite de score : TrustMark a souvent plusieurs boites a 1,00 autour de la vraie) ;
  --regime   : un seuil de probabilite par regime (t_full, t_vide) au lieu d'un seul ;
  --menu     : regime vide : carte seuillee | meilleur rectangle ajuste sur la carte | tout a 0 ;
               regime full : carte seuillee | tout a 1 ;
  --net      : nettoyage : petits ilots supprimes (trous < a0 cases en regime full, taches < a1 cases en regime vide) ;
  --oof ...  : une ou plusieurs sorties de U-Net (moyenne des probabilites) comme carte de repli.
  python regle60.py --oof <a.npz> [<b.npz> ...] [--cons] [--regime] [--menu] [--net] [--ecrire subs/nom.pt]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.dont_write_bytecode = True
T = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
W = Path("/p/scratch/training2666/huang25/Epi84/depot-coequipier")
V = Path(__file__).resolve().parent
METHODES = ["pixelseal", "trustmark", "stegastamp"]; K = 5
T_FULL = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
T_BOITE = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
T_C = [round(float(t), 2) for t in np.linspace(0.1, 0.9, 17)]
T_R = [0.3, 0.4, 0.5, 0.6, 0.7]
AIRES = [0, 8, 16, 32, 64, 128]


def miou_lot(B, G):
    B = np.asarray(B).astype(bool); G = np.asarray(G).astype(bool); s = np.zeros(len(B))
    for b, g in ((B, G), (~B, ~G)):
        inter = (b & g).sum((1, 2)); union = (b | g).sum((1, 2))
        s += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return s / 2


def masque_boite(b):
    m = np.zeros((512, 512), np.float32); y0, y1, x0, x1 = [int(round(float(v))) for v in b[:4]]; m[y0:y1, x0:x1] = 1
    return m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5


def charger_boites(split):
    R = {}
    for f in sorted((T / "output/rect").glob(f"{split}_*.npz")):
        z = np.load(f)
        for n, b, s in zip(z["names"].tolist(), z["boites"], z["full"]):
            R[n] = (b, float(s))
    return R


def consensus(b):
    top = b[b[:, 4] >= b[0, 4] - 1e-6]
    return np.median(top[:, :4], axis=0)


def meilleur_rect(g):
    """Rectangle de somme maximale dans la grille de gains g (Kadane 2D). Renvoie (somme, (y0, y1, x0, x1))."""
    n, m = g.shape; best, arg = -np.inf, (0, 0, 0, 0)
    cs = np.vstack([np.zeros((1, m)), np.cumsum(g, 0)])
    for top in range(n):
        col = cs[top + 1:] - cs[top]
        P = np.concatenate([np.zeros((col.shape[0], 1)), np.cumsum(col, 1)], 1)
        vals = P[:, 1:] - np.minimum.accumulate(P[:, :-1], 1)
        b, r = np.unravel_index(np.argmax(vals), vals.shape)
        if vals[b, r] > best:
            best = vals[b, r]; arg = (top, top + b + 1, int(np.argmin(P[b, :r + 1])), r + 1)
    return best, arg


def masques_rect(C, t):
    """Pour chaque carte : le rectangle qui maximise la somme de (proba - t) ; vide si aucune somme positive."""
    out = np.zeros(C.shape, bool)
    for i, c in enumerate(C):
        s, (y0, y1, x0, x1) = meilleur_rect(c.astype(np.float64) - t)
        if s > 0:
            out[i, y0:y1, x0:x1] = True
    return out


class D:
    pass


CLES = ("meths", "C", "Bb0", "Bb1", "sb", "sf", "G", "pi", "full")


def charger(split, oofs, menu):
    meta = json.loads((T / "release" / split / "metadata.json").read_text())
    R = charger_boites(split); d = D()
    d.noms = [m["image_name"] for m in meta]; d.meths = np.array([m["method"] for m in meta])
    C = 0
    for f in oofs:
        z = np.load(f); kn, kp = ("names", "probs") if split == "train" else ("test_names", "test_probs")
        P = dict(zip(z[kn].tolist(), z[kp])); C = C + np.stack([P[n] for n in d.noms]).astype(np.float32)
    d.C = C / len(oofs)
    d.Bb0 = np.stack([masque_boite(R[n][0][0]) for n in d.noms])
    d.Bb1 = np.stack([masque_boite(consensus(R[n][0])) for n in d.noms])
    d.sb = np.array([R[n][0][0][4] for n in d.noms]); d.sf = np.array([R[n][1] for n in d.noms])
    d.G = d.pi = d.full = None
    d.Rect = {t: masques_rect(d.C, t) for t in T_R} if menu else {}
    if split == "train":
        z = np.load(T / "release/train/masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
        d.G = np.stack([M[n] for n in d.noms]).astype(bool)
        plis = json.loads((W / "watermark/splits.json").read_text())["plis"]; d.pi = np.array([plis[n] for n in d.noms])
        d.full = np.array([bool(m["full_watermark"]) for m in meta])
    return d


def sous(d, sel):
    e = D(); e.noms = [n for n, s in zip(d.noms, sel) if s]
    for k in CLES:
        v = getattr(d, k); setattr(e, k, None if v is None else v[sel])
    e.Rect = {t: v[sel] for t, v in d.Rect.items()}
    return e


def nettoyer(P, a0, a1):
    """Supprime les petits ilots : composantes de 0 de moins de a0 cases -> 1 ; composantes de 1 de moins de a1 -> 0."""
    if a0 <= 0 and a1 <= 0:
        return P
    P = P.copy()
    for i in range(len(P)):
        for val, a in ((False, a0), (True, a1)):
            if a > 0:
                lab, n = ndimage.label(P[i] == val)
                if n:
                    t = np.bincount(lab.ravel()); petit = np.where(t < a)[0]; petit = petit[petit > 0]
                    if len(petit):
                        P[i][np.isin(lab, petit)] = not val
    return P


def predire(d, p):
    full = d.sf >= p["tf"]; boite = ~full & (d.sb >= p["tb"]); vide = ~full & ~boite
    P = np.zeros(d.C.shape, bool)
    if p.get("m_full", "carte") == "carte":
        P[full] = nettoyer(d.C[full] >= p["t_full"], p.get("a0", 0), 0)
    else:
        P[full] = True
    mv = p.get("m_vide", "carte")
    if mv == "carte":
        P[vide] = nettoyer(d.C[vide] >= p["t_vide"], 0, p.get("a1", 0))
    elif mv.startswith("rect"):
        P[vide] = d.Rect[float(mv[4:])][vide]
    P[boite] = (d.Bb1 if p.get("cons") else d.Bb0)[boite]
    return P


def regler(d, o):
    sc = lambda p: miou_lot(predire(d, p), d.G).mean()
    t = float(max(T_C, key=lambda t: miou_lot(d.C >= t, d.G).mean()))
    p = {"t_full": t, "t_vide": t, "tf": 1.01, "tb": 1.01}
    paires = [(a, b) for a in T_FULL for b in T_BOITE]
    tf, tb = max(paires, key=lambda q: sc({**p, "tf": q[0], "tb": q[1]})); p.update(tf=tf, tb=tb)
    if o.cons:
        p["cons"] = int(max((0, 1), key=lambda c: sc({**p, "cons": c})))
    if o.regime:
        for _ in range(2):
            p["t_full"] = float(max(T_C, key=lambda t: sc({**p, "t_full": t})))
            p["t_vide"] = float(max(T_C, key=lambda t: sc({**p, "t_vide": t})))
            tf, tb = max(paires, key=lambda q: sc({**p, "tf": q[0], "tb": q[1]})); p.update(tf=tf, tb=tb)
    if o.menu:
        p["m_vide"] = max(["carte"] + [f"rect{t}" for t in T_R] + ["zeros"], key=lambda v: sc({**p, "m_vide": v}))
        p["m_full"] = max(["carte", "uns"], key=lambda v: sc({**p, "m_full": v}))
    if o.net:
        p["a0"] = int(max(AIRES, key=lambda a: sc({**p, "a0": a})))
        p["a1"] = int(max(AIRES, key=lambda a: sc({**p, "a1": a})))
    return p


def evaluer(d, o):
    sc = np.zeros(len(d.noms)); choix = []
    for m in METHODES:
        for k in range(K):
            tr = (d.meths == m) & (d.pi != k); te = (d.meths == m) & (d.pi == k)
            p = regler(sous(d, tr), o)
            sc[te] = miou_lot(predire(sous(d, te), p), d.G[te]); choix.append((m, k, p))
    return sc, choix


def texte(sc, d):
    pp = np.array([sc[d.pi == k].mean() for k in range(K)])
    s = f"{pp.mean():.4f} ± {pp.std():.4f}  (" + "  ".join(f"{m[:5]} {sc[d.meths == m].mean():.4f}" for m in METHODES) + ")"
    s += "   [full/partiel : " + "  ".join(f"{m[:2]} {sc[(d.meths == m) & d.full].mean():.3f}/{sc[(d.meths == m) & ~d.full].mean():.3f}" for m in METHODES) + "]"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", default=[str(T / "output/unet_h5a_feats_oof.npz")])
    ap.add_argument("--cons", action="store_true"); ap.add_argument("--regime", action="store_true")
    ap.add_argument("--menu", action="store_true"); ap.add_argument("--net", action="store_true")
    ap.add_argument("--ecrire", default=None); ap.add_argument("--choix", action="store_true")
    ap.add_argument("--ref", default=None, help="scores par image de reference (npy) pour afficher le gain par pli")
    ap.add_argument("--sauver", default=None, help="enregistrer les scores par image (npy)")
    a = ap.parse_args()
    d = charger("train", a.oof, a.menu)
    sc, choix = evaluer(d, a)
    nom = f"cons={int(a.cons)} regime={int(a.regime)} menu={int(a.menu)} net={int(a.net)} oof={[Path(f).name[:22] for f in a.oof]}"
    print(f"{nom}\n   PROTOCOLE : {texte(sc, d)}", flush=True)
    if a.choix:
        for m, k, p in choix:
            print(f"      {m[:5]} pli {k} : {p}")
    if a.sauver:
        np.save(a.sauver, sc)
    if a.ref:
        r = np.load(a.ref); g = np.array([(sc - r)[d.pi == k].mean() for k in range(K)])
        print(f"   gain / ref : {g.mean():+.4f} ± {g.std():.4f}, plis positifs {(g > 0).sum()}/5, par methode "
              + "  ".join(f"{m[:5]} {(sc - r)[d.meths == m].mean():+.4f}" for m in METHODES), flush=True)
    if a.ecrire:
        import torch
        dt = charger("test", a.oof, a.menu)
        params = {m: regler(sous(d, d.meths == m), a) for m in METHODES}
        P = np.zeros(dt.C.shape, bool)
        for m in METHODES:
            sel = dt.meths == m; P[sel] = predire(sous(dt, sel), params[m])
        assert P.shape == (1020, 128, 128) and len(set(dt.noms)) == 1020
        out = V / a.ecrire
        torch.save({"image_names": list(dt.noms), "masks": torch.from_numpy(P).bool()}, out)
        Path(str(out).replace(".pt", "_params.json")).write_text(json.dumps(
            {"options": {"cons": a.cons, "regime": a.regime, "menu": a.menu, "net": a.net, "oof": a.oof},
             "params": params, "score_protocole": float(np.mean([sc[d.pi == k].mean() for k in range(K)])),
             "par_methode": {m: float(sc[d.meths == m].mean()) for m in METHODES}}, indent=1))
        print("ecrit", out, P.shape, "part de 1 :", round(float(P.mean()), 4), "params :", params, flush=True)
        for m in METHODES:
            sel = dt.meths == m; p = params[m]
            full = dt.sf[sel] >= p["tf"]; boite = ~full & (dt.sb[sel] >= p["tb"])
            ftr = d.sf[d.meths == m] >= p["tf"]; btr = ~ftr & (d.sb[d.meths == m] >= p["tb"])
            print(f"   {m} : regimes test full {full.mean():.2f} boite {boite.mean():.2f} vide {(~full & ~boite).mean():.2f}"
                  f"   | train full {ftr.mean():.2f} boite {btr.mean():.2f} vide {(~ftr & ~btr).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
