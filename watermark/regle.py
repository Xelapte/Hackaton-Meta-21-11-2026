"""Tâche 1, H3c : règle de décision étendue sur la recherche de rectangle, et prédiction finale du test.

Régimes (par image, méthode connue) :
  - « full »          : précision de l'image entière >= t_full ;
  - « partiel boîte » : sinon, meilleure boîte >= t_boite ;
  - « partiel vide »  : sinon.
Menus discrets (choisis par méthode, chacun sur les images de son régime, dans les plis d'entraînement) :
  - full          : carte (seuil H1) | tout à 1 | trous = cases de cote z < -z dans l'image, z ∈ {1 ; 1,5 ; 2 ; 2,5} (A)
  - partiel boîte : boîte | boîte ∩ carte (B)
  - partiel vide  : carte | tout à 0
H3b = (carte, boîte, carte). Le réglage se fait dans l'ordre : seuil de carte, (t_full, t_boite), puis les menus.
Score : réglage sur 4 plis, mesure sur le 5e. --predire : réglage sur tout le train, application au test.

  python watermark/regle.py [--repli unet]     # évaluation H3b (plis gelés) ; --menus : H3c et ablations
  python watermark/regle.py --predire          # + soumission du test (si les recherches du test existent)
  python watermark/regle.py --fake
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.protocole import K, METHODES, charger_plis, gain_robuste, miou_lot, resume, texte  # noqa: E402

CANAL = {"pixelseal": "ps_512", "trustmark": "tm_245", "stegastamp": "ss_16"}
T_FULL = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
T_BOITE = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
MENU_FULL = ["carte", "uns", "z1", "z1.5", "z2", "z2.5"]
MENU_BOITE = ["boite", "boite∩carte"]
MENU_VIDE = ["carte", "zeros"]
H3B = {"full": "carte", "boite": "boite", "vide": "carte"}


def masque_boite(b):
    m = np.zeros((512, 512), np.float32); y0, y1, x0, x1 = [int(v) for v in b[:4]]; m[y0:y1, x0:x1] = 1
    return m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5


class Donnees:
    """C (cartes), Bb (masque de la meilleure boîte), sb (son score), sf (image entière), meths [, G, pi]."""
    def __init__(self, noms, meths, C, Bb, sb, sf, G=None, pi=None):
        self.noms, self.meths, self.C, self.Bb, self.sb, self.sf, self.G, self.pi = noms, meths, C, Bb, sb, sf, G, pi

    def sous(self, sel):
        return Donnees([n for n, s in zip(self.noms, sel) if s], self.meths[sel], self.C[sel], self.Bb[sel],
                       self.sb[sel], self.sf[sel], None if self.G is None else self.G[sel],
                       None if self.pi is None else self.pi[sel])


def predire(D, p):
    """p : dict t_carte, tf, tb, full, boite, vide. Renvoie N x 128 x 128 bool."""
    C = D.C; carte = C >= p["t_carte"]
    full = D.sf >= p["tf"]; boite = ~full & (D.sb >= p["tb"]); vide = ~full & ~boite
    P = np.zeros(C.shape, bool)
    # full
    if p["full"] == "carte":
        P[full] = carte[full]
    elif p["full"] == "uns":
        P[full] = True
    else:
        z0 = float(p["full"][1:]); c = C[full]
        z = (c - c.mean((1, 2), keepdims=True)) / (c.std((1, 2), keepdims=True) + 1e-6)
        P[full] = ~(z < -z0)
    # partiel avec boîte
    if p["boite"] == "boite":
        P[boite] = D.Bb[boite]
    else:
        inter = D.Bb[boite] & carte[boite]
        vides = inter.reshape(len(inter), -1).sum(1) == 0
        inter[vides] = D.Bb[boite][vides]
        P[boite] = inter
    # partiel sans boîte
    P[vide] = carte[vide] if p["vide"] == "carte" else False
    return P


def regler(D, menus=True):
    """Réglage sur D (une méthode, plis d'entraînement). menus=False : H3b."""
    cand = np.quantile(D.C, np.linspace(0.02, 0.98, 49))
    t_carte = float(max(cand, key=lambda t: miou_lot(D.C >= t, D.G).mean()))
    p = {"t_carte": t_carte, **H3B}
    tf, tb = max(((a, b) for a in T_FULL for b in T_BOITE),
                 key=lambda q: miou_lot(predire(D, {**p, "tf": q[0], "tb": q[1]}), D.G).mean())
    p.update(tf=tf, tb=tb)
    if menus:
        full = D.sf >= tf; boite = ~full & (D.sb >= tb); vide = ~full & ~boite
        for cle, menu, sel in (("full", MENU_FULL, full), ("boite", MENU_BOITE, boite), ("vide", MENU_VIDE, vide)):
            if menus is not True and cle not in menus:
                continue
            if sel.sum() == 0:
                continue
            p[cle] = max(menu, key=lambda o: miou_lot(predire(D, {**p, cle: o})[sel], D.G[sel]).mean())
    return p


def evaluer_imbrique(D, menus):
    sc = np.zeros(len(D.noms)); choix = []
    for m in METHODES:
        for k in range(K):
            tr = (D.meths == m) & (D.pi != k); te = (D.meths == m) & (D.pi == k)
            if not te.any():
                continue
            p = regler(D.sous(tr), menus)
            sc[te] = miou_lot(predire(D.sous(te), p), D.G[te]); choix.append((m, k, p))
    return sc, choix


def charger(args, split):
    off = Path(args.out_dir)
    meta = json.loads((args.data / split / "metadata.json").read_text())
    R = {}
    dossiers = ["rect"] + (["rect_fin"] if getattr(args, "fin", False) else [])
    for dos in dossiers:                         # meilleure boîte des recherches grossière et fine
        for f in sorted((off / dos).glob(f"{split}_*.npz")):
            z = np.load(f)
            for n, b, s in zip(z["names"].tolist(), z["boites"], z["full"]):
                if n not in R or b[0][4] > R[n][0][4]:
                    R[n] = (b[0], float(s) if n not in R else R[n][1])
    meta = [m for m in meta if m["image_name"] in R]
    noms = [m["image_name"] for m in meta]; meths = np.array([m["method"] for m in meta])
    if args.repli == "unet":                     # probabilités hors pli du U-Net (train) / moyenne des 5 modèles (test)
        zo = np.load(args.oof)
        cle_n, cle_p = ("names", "probs") if split == "train" else ("test_names", "test_probs")
        Pm = dict(zip(zo[cle_n].tolist(), zo[cle_p]))
        C = np.stack([Pm[n] for n in noms]).astype(np.float32)
    else:
        zc = np.load(args.cartes); Fm = dict(zip(zc["names"].tolist(), zc["feats"])); can = zc["canaux"].tolist()
        C = np.stack([Fm[n][can.index(CANAL[m])] for n, m in zip(noms, meths)]).astype(np.float32)
    Bb = np.stack([masque_boite(R[n][0]) for n in noms]); sb = np.array([R[n][0][4] for n in noms])
    sf = np.array([R[n][1] for n in noms])
    G = pi = None
    if split == "train":
        z = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
        G = np.stack([M[n] for n in noms]).astype(bool); plis = charger_plis(); pi = np.array([plis[n] for n in noms])
    return Donnees(noms, meths, C, Bb, sb, sf, G, pi), meta


def faux():
    rng = np.random.default_rng(0); n = 60
    meths = np.array([METHODES[i % 3] for i in range(n)]); G = np.zeros((n, 128, 128), bool)
    for i in range(n):
        y, x = rng.integers(0, 90, 2); G[i, y:y + 30, x:x + 30] = True
        if i % 2:
            G[i] = ~G[i]
    C = G + rng.normal(0, 1, G.shape).astype(np.float32)
    Bb = np.roll(G, 3, 1); sb = rng.uniform(.5, 1, n); sf = np.where(np.arange(n) % 2, .95, .5)
    return Donnees([f"i{i}" for i in range(n)], meths, C.astype(np.float32), Bb, sb, sf, G, np.arange(n) % K)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--cartes", default=str(off / "output" / "cartes_feats.npz"))
    ap.add_argument("--repli", choices=["carte", "unet"], default="carte", help="prédiction de repli (hors boîte)")
    ap.add_argument("--oof", default=str(off / "output" / "unet_h5a_feats_oof.npz"), help="--repli unet : sorties du U-Net")
    ap.add_argument("--fin", action="store_true", help="H8b : ajouter la recherche fine (output/rect_fin)")
    ap.add_argument("--menus", action="store_true", help="H3c : menus étendus (sans gain démontré, 21/11 19h10)")
    ap.add_argument("--predire", action="store_true")
    ap.add_argument("--sortie", default=None, help="soumission (défaut : <out_dir>/regle_h3c.pt)")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    D = faux() if args.fake else charger(args, "train")[0]
    print(f"{len(D.noms)} images du train ({ {m: int((D.meths == m).sum()) for m in METHODES} })", flush=True)
    s_b, choix_b = evaluer_imbrique(D, menus=False)
    print(f"H3b (rectangle + repli {args.repli}) : {texte(resume(s_b, D.pi, D.meths))}", flush=True)
    if args.repli == "unet":                     # comparaison au repli carte, sur les mêmes images
        a2 = argparse.Namespace(**{**vars(args), "repli": "carte"})
        s_carte, _ = evaluer_imbrique(charger(a2, "train")[0], menus=False)
        print(f"H3b (repli carte)               : {texte(resume(s_carte, D.pi, D.meths))}", flush=True)
        print(f"gain repli unet / carte : {gain_robuste(s_carte, s_b, D.pi)}", flush=True)
    if args.fin:                                 # comparaison à H3b sans la recherche fine, mêmes images
        a3 = argparse.Namespace(**{**vars(args), "fin": False})
        s_sans, _ = evaluer_imbrique(charger(a3, "train")[0], menus=False)
        print(f"H3b sans recherche fine         : {texte(resume(s_sans, D.pi, D.meths))}", flush=True)
        print(f"gain recherche fine (H8b) : {gain_robuste(s_sans, s_b, D.pi)}", flush=True)
    s_c, choix = (s_b, choix_b)
    if args.menus:
        s_c, choix = evaluer_imbrique(D, menus=True)
        print(f"H3c (règle étendue)            : {texte(resume(s_c, D.pi, D.meths))}", flush=True)
        print(f"gain H3c / H3b : {gain_robuste(s_b, s_c, D.pi)}", flush=True)
    for cle in (("full", "boite", "vide") if args.menus else ()):   # ablations (pour le journal, pas pour choisir)
        s_a, _ = evaluer_imbrique(D, menus=[cle])
        g = gain_robuste(s_b, s_a, D.pi)
        print(f"  ablation, menu « {cle} » seul : {s_a.mean():.4f} (gain {g['gain']:+.4f} ± {g['ecart_type']:.4f}, "
              f"{g['plis_positifs']}/5 plis)", flush=True)
    for m in METHODES:
        cs = [(c["full"], c["boite"], c["vide"], c["tf"], c["tb"]) for mm, k, c in choix if mm == m]
        print(f"  choix par pli, {m} (full, boîte, vide, t_full, t_boite) : {cs}", flush=True)
    if args.predire and not args.fake:
        Dt, meta_t = charger(args, "test")
        tous = json.loads((args.data / "test" / "metadata.json").read_text())
        manque = {m["method"] for m in tous if m["image_name"] not in set(Dt.noms)}
        if manque:
            sys.exit(f"Recherche de rectangle manquante sur le test pour : {sorted(manque)}")
        params = {m: regler(D.sous(D.meths == m), args.menus) for m in METHODES}
        P = np.zeros(Dt.C.shape, bool)
        for m in METHODES:
            sel = Dt.meths == m; P[sel] = predire(Dt.sous(sel), params[m])
        sys.path.insert(0, str(RACINE))
        from watermark.baseline import ecrire_soumission, journal
        sortie = args.sortie or str(Path(args.out_dir) / f"regle_h3b_{args.repli}{'_fin' if args.fin else ''}{'_menus' if args.menus else ''}.pt")
        ecrire_soumission([{"image_name": n} for n in Dt.noms], list(P.astype(np.uint8)), sortie)
        Path(sortie.replace(".pt", "_params.json")).write_text(json.dumps(params, indent=1, ensure_ascii=False))
        print("paramètres (réglés sur tout le train) :", params, flush=True)
        journal({"mode": "regle_h3b", "repli": args.repli, "fin": args.fin, "menus": args.menus, "params": params,
                 "protocole": "plis gelés, imbriqué"}, float(s_c.mean()))


if __name__ == "__main__":
    main()
