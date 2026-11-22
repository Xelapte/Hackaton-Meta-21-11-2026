"""v60 / decision « mIoU attendu maximal » par image (CPU leger), evaluee sur les plis geles.

Idee : la metrique est, par image, la moyenne de l'IoU de la classe 1 et de l'IoU de la classe 0. La petite
classe (un trou dans une image full, une petite zone tatouee dans une image partielle) pese donc autant que la
grande. Un seuil fixe a 0,5 sur les probabilites oublie presque toujours la petite classe.
Ici, pour chaque image, on trie les cases par probabilite et on choisit le NOMBRE k de cases mises a 1 qui
maximise le mIoU attendu si les probabilites sont justes :
   IoU1(k) ~ S_k / (k + S_N - S_k)      (S_k = somme des k plus grandes probabilites)
   IoU0(k) ~ Z_k / (N - k + k - S_k)    (Z_k = somme des (1 - p) hors des k premieres)
Les probabilites sont d'abord recalibrees par methode : p' = sigmoide(a * logit(p) + b) ; (a, b) regles dans les
plis d'entrainement (2 parametres par methode, ou 2 par regime full / non full avec --regime).
  python decide.py --oof unet2_c1_oof.npz [--regime] [--ecrire subs/nom.pt]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
import regle60 as R  # noqa: E402

A_ = [0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
B_ = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
T_FULL = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]


def masques_emiou(P, a, b):
    """P : N x 128 x 128 probabilites. Renvoie le masque (bool) qui maximise le mIoU attendu, image par image."""
    n = len(P); flat = np.clip(P.reshape(n, -1).astype(np.float64), 1e-4, 1 - 1e-4)
    q = 1 / (1 + np.exp(-(a * np.log(flat / (1 - flat)) + b)))
    ordre = np.argsort(-q, axis=1); sp = np.take_along_axis(q, ordre, 1); N = q.shape[1]
    S = np.concatenate([np.zeros((n, 1)), np.cumsum(sp, 1)], 1); SN = S[:, -1:]; k = np.arange(N + 1)[None]
    iou1 = S / np.maximum(k + SN - S, 1e-9)
    iou0 = ((N - k) - (SN - S)) / np.maximum((N - k) + (k - S), 1e-9)
    iou1[:, 0] = np.exp(np.log1p(-q).sum(1))            # rien predit : IoU1 = 1 seulement si la verite est vide
    iou0[:, N] = np.exp(np.log(q).sum(1))
    kb = np.argmax(iou1 + iou0, axis=1)
    rang = np.empty_like(ordre); np.put_along_axis(rang, ordre, np.broadcast_to(np.arange(N), ordre.shape), 1)
    return (rang < kb[:, None]).reshape(P.shape)


def predire(d, p, cache=None):
    if "tf" not in p:
        return masques_emiou(d.C, p["a"], p["b"])
    full = d.sf >= p["tf"]; P = np.zeros(d.C.shape, bool)
    if full.any():
        P[full] = masques_emiou(d.C[full], p["a"], p["b"])
    if (~full).any():
        P[~full] = masques_emiou(d.C[~full], p["a2"], p["b2"])
    return P


def regler(d, regime):
    sc = lambda p: R.miou_lot(predire(d, p), d.G).mean()
    a, b = max(((a, b) for a in A_ for b in B_), key=lambda q: sc({"a": q[0], "b": q[1]}))
    p = {"a": a, "b": b}
    if regime:
        best = (-1, None)
        for tf in T_FULL:
            full = d.sf >= tf
            if full.sum() < 5 or (~full).sum() < 5:
                continue
            df, dv = R.sous(d, full), R.sous(d, ~full)
            qf = max(((a, b) for a in A_ for b in B_), key=lambda q: R.miou_lot(masques_emiou(df.C, *q), df.G).mean())
            qv = max(((a, b) for a in A_ for b in B_), key=lambda q: R.miou_lot(masques_emiou(dv.C, *q), dv.G).mean())
            cand = {"tf": tf, "a": qf[0], "b": qf[1], "a2": qv[0], "b2": qv[1]}; s = sc(cand)
            if s > best[0]:
                best = (s, cand)
        if best[1] is not None and best[0] > sc(p):
            p = best[1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True); ap.add_argument("--regime", action="store_true")
    ap.add_argument("--ecrire", default=None); ap.add_argument("--ref", default=None); ap.add_argument("--sauver", default=None)
    ap.add_argument("--choix", action="store_true")
    a = ap.parse_args()
    d = R.charger("train", a.oof, False)
    sc = np.zeros(len(d.noms)); choix = []
    for m in R.METHODES:
        for k in range(R.K):
            tr = (d.meths == m) & (d.pi != k); te = (d.meths == m) & (d.pi == k)
            p = regler(R.sous(d, tr), a.regime); choix.append((m, k, p))
            sc[te] = R.miou_lot(predire(R.sous(d, te), p), d.G[te])
    print(f"emiou regime={int(a.regime)} oof={[Path(f).name[:22] for f in a.oof]}\n   PROTOCOLE : {R.texte(sc, d)}", flush=True)
    if a.choix:
        for m, k, p in choix:
            print(f"      {m[:5]} pli {k} : {p}")
    if a.sauver:
        np.save(a.sauver, sc)
    if a.ref:
        r = np.load(a.ref); g = np.array([(sc - r)[d.pi == k].mean() for k in range(R.K)])
        print(f"   gain / ref : {g.mean():+.4f} ± {g.std():.4f}, plis positifs {(g > 0).sum()}/5, par methode "
              + "  ".join(f"{m[:5]} {(sc - r)[d.meths == m].mean():+.4f}" for m in R.METHODES), flush=True)
    if a.ecrire:
        import torch
        dt = R.charger("test", a.oof, False)
        params = {m: regler(R.sous(d, d.meths == m), a.regime) for m in R.METHODES}
        P = np.zeros(dt.C.shape, bool)
        for m in R.METHODES:
            sel = dt.meths == m; P[sel] = predire(R.sous(dt, sel), params[m])
        assert P.shape == (1020, 128, 128) and len(set(dt.noms)) == 1020
        out = V / a.ecrire
        torch.save({"image_names": list(dt.noms), "masks": torch.from_numpy(P).bool()}, out)
        Path(str(out).replace(".pt", "_params.json")).write_text(json.dumps(
            {"decision": "emiou", "options": {"regime": a.regime, "oof": a.oof}, "params": params,
             "score_protocole": float(np.mean([sc[d.pi == k].mean() for k in range(R.K)])),
             "par_methode": {m: float(sc[d.meths == m].mean()) for m in R.METHODES}}, indent=1))
        print("ecrit", out, P.shape, "part de 1 :", round(float(P.mean()), 4), "params :", params, flush=True)


if __name__ == "__main__":
    main()
