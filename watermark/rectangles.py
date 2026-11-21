"""Tâche 1, H3b : recherche du rectangle tatoué par décodage (message connu), puis règle de décision.

Constat (H3a) : décoder exactement la boîte de la zone tatouée donne une précision de bits bien au-dessus du
hasard (TrustMark médiane 0,95), et une boîte décalée de 25 % retombe au hasard. On cherche donc, pour chaque
image, la boîte (rapport <= 16:1, <= 2:1 pour TrustMark) qui maximise la précision de bits :
  1. grille grossière : tailles {48..512} x {48..512} (rapport permis), pas = moitié de la taille ;
  2. raffinement autour des --top meilleures : décalages de +-1/8 et +-1/4, échelles 0,75 / 1 / 1,33.
Sortie (--recherche) : <out_dir>/rect/<split>_<méthode>[_<part>].npz : pour chaque image, les --garder meilleures
boîtes (y0, y1, x0, x1, précision) et la précision de l'image entière.

Règle (--evaluer, CPU) : si la précision de l'image entière >= t_full, régime « full » -> prédiction de repli ;
sinon, si la meilleure boîte >= t_boite -> masque = cette boîte ; sinon -> repli. Repli = meilleure carte de H1
(seuil par méthode). t_full, t_boite : par méthode, réglés sur 4 plis, mesurés sur le 5e (2 paramètres par méthode).

  bash slurm/soumettre.sh 2 wm_rect_ps --venv watermark python watermark/rectangles.py --recherche --methode pixelseal
  python watermark/rectangles.py --evaluer
  python watermark/rectangles.py --recherche --methode trustmark --fake
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.protocole import K, METHODES, charger_plis, gain_robuste, miou_lot, resume, texte  # noqa: E402

TAILLES = [48, 64, 96, 128, 192, 256, 384, 512]
RAPPORT_MAX = {"pixelseal": 16, "trustmark": 2, "stegastamp": 16}


class FauxTatouage:
    nbits = 100
    def logits(self, x):
        return torch.randn(len(x), self.nbits)


def grille(meth):
    boites = []
    for h in TAILLES:
        for w in TAILLES:
            if max(h, w) / min(h, w) > RAPPORT_MAX[meth]:
                continue
            sy, sx = max(16, h // 2), max(16, w // 2)
            ys = sorted(set(list(range(0, 512 - h + 1, sy)) + [512 - h]))
            xs = sorted(set(list(range(0, 512 - w + 1, sx)) + [512 - w]))
            boites += [(y, y + h, x, x + w) for y in ys for x in xs]
    return boites


def raffiner(b, meth):
    y0, y1, x0, x1 = b; h, w = y1 - y0, x1 - x0; out = set()
    for eh in (0.75, 1.0, 1.33):
        for ew in (0.75, 1.0, 1.33):
            hh, ww = int(min(512, max(32, h * eh))), int(min(512, max(32, w * ew)))
            if max(hh, ww) / min(hh, ww) > RAPPORT_MAX[meth]:
                continue
            for dy in (-0.25, -0.125, 0, 0.125, 0.25):
                for dx in (-0.25, -0.125, 0, 0.125, 0.25):
                    cy = (y0 + y1) / 2 + dy * h; cx = (x0 + x1) / 2 + dx * w
                    ny = int(np.clip(cy - hh / 2, 0, 512 - hh)); nx = int(np.clip(cx - ww / 2, 0, 512 - ww))
                    out.add((ny, ny + hh, nx, nx + ww))
    return list(out)


@torch.no_grad()
def precision_boites(wm, x, boites, signe, bs):
    """x : 3 x 512 x 512 (GPU, [0,1]) ; renvoie la précision de bits de chaque boîte (regroupées par taille)."""
    res = np.zeros(len(boites), np.float32)
    groupes = {}
    for i, (y0, y1, x0, x1) in enumerate(boites):
        groupes.setdefault((y1 - y0, x1 - x0), []).append(i)
    for (h, w), idx in groupes.items():
        for j in range(0, len(idx), bs):
            lot = idx[j:j + bs]
            crops = torch.stack([x[:, boites[i][0]:boites[i][1], boites[i][2]:boites[i][3]] for i in lot])
            lg = wm.logits(crops).float().to(signe.device)
            res[lot] = ((lg * signe[None]) > 0).float().mean(1).cpu().numpy()
    return res


def recherche(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    if args.fake:
        wm = FauxTatouage(); meta = [{"image_name": f"i{k}", "method": args.methode,
                                       "message": "".join(rng.choice(["0", "1"], 100))} for k in range(2)]
        charger = lambda n: rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        out_dir = Path("/tmp/wm_fake_rect")
    else:
        sys.path.insert(0, str(args.tache))
        from wmloc.watermarks import load_watermark
        wm = load_watermark(args.methode, models_dir=args.tache / "models", device=dev)
        meta = [m for m in json.loads((args.data / args.split / "metadata.json").read_text()) if m["method"] == args.methode]
        charger = lambda n: np.asarray(Image.open(args.data / args.split / "images" / f"{n}.png").convert("RGB"))
        out_dir = Path(args.out_dir)
    if args.tailles:
        global TAILLES
        TAILLES = [int(t) for t in args.tailles.split(",")]
    if args.parts > 1:
        meta = meta[args.part::args.parts]
    if args.max_images:
        meta = meta[:args.max_images]
    (out_dir / "rect").mkdir(parents=True, exist_ok=True)
    g = grille(args.methode)
    print(f"{args.methode} / {args.split} : {len(meta)} images, {len(g)} boîtes grossières par image", flush=True)
    noms, boites_sortie, full = [], [], []
    t0 = time.time()
    for n, m in enumerate(meta):
        x = torch.from_numpy(charger(m["image_name"])).permute(2, 0, 1).float().div(255).to(dev)
        signe = torch.tensor([int(c) * 2 - 1 for c in m["message"]][:wm.nbits], dtype=torch.float32, device=dev)
        p = precision_boites(wm, x, g, signe, args.bs)
        top = [g[i] for i in np.argsort(-p)[:args.top]]
        fin = list({b for t in top for b in raffiner(t, args.methode)})
        pf = precision_boites(wm, x, fin, signe, args.bs)
        tout = g + fin; pt = np.concatenate([p, pf])
        ordre = np.argsort(-pt)
        garde, vus = [], set()
        for i in ordre:
            if tout[i] not in vus:
                vus.add(tout[i]); garde.append((*tout[i], pt[i]))
            if len(garde) >= args.garder:
                break
        noms.append(m["image_name"]); boites_sortie.append(np.array(garde, np.float32))
        full.append(float(precision_boites(wm, x, [(0, 512, 0, 512)], signe, 1)[0]))
        if n % 20 == 0 or n == len(meta) - 1:
            print(f"  {n + 1}/{len(meta)}  meilleure {garde[0][4]:.3f}  image entière {full[-1]:.3f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    suffixe = f"_{args.part}" if args.parts > 1 else ""
    chemin = out_dir / "rect" / f"{args.split}_{args.methode}{suffixe}.npz"
    np.savez(chemin, names=np.array(noms), boites=np.stack(boites_sortie), full=np.array(full, np.float32))
    print(f"Écrit : {chemin}", flush=True)


def masque_boite(b):
    m = np.zeros((512, 512), np.float32); y0, y1, x0, x1 = [int(v) for v in b[:4]]; m[y0:y1, x0:x1] = 1
    return (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5)


def charger_rect(dossier, split):
    R = {}
    for f in sorted(Path(dossier).glob(f"{split}_*.npz")):
        z = np.load(f)
        for n, b, s in zip(z["names"].tolist(), z["boites"], z["full"]):
            R[n] = (b, float(s))
    return R


def evaluer(args):
    off_out = Path(args.out_dir)
    meta = json.loads((args.data / "train" / "metadata.json").read_text())
    z = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
    plis = charger_plis()
    R = charger_rect(off_out / "rect", "train")
    meta = [m for m in meta if m["image_name"] in R]
    noms = [m["image_name"] for m in meta]; meths = np.array([m["method"] for m in meta])
    pi = np.array([plis[n] for n in noms]); G = np.stack([M[n] for n in noms]).astype(bool)
    print(f"{len(noms)} images du train avec recherche de rectangle", flush=True)
    # repli : meilleure carte de H1 (seuil par méthode réglé dans les plis)
    zc = np.load(args.cartes); Fm = dict(zip(zc["names"].tolist(), zc["feats"])); can = zc["canaux"].tolist()
    repli_canal = {"pixelseal": "ps_512", "trustmark": "tm_245", "stegastamp": "ss_16"}
    C = np.stack([Fm[n][can.index(repli_canal[m["method"]])] for n, m in zip(noms, meta)]).astype(np.float32)
    Bbox = np.stack([masque_boite(R[n][0][0]) for n in noms])
    sbox = np.array([R[n][0][0][4] for n in noms]); sfull = np.array([R[n][1] for n in noms])
    T_FULL = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
    T_BOITE = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]

    def predire(sel, t_carte, tf, tb):
        P = C[sel] >= t_carte
        use = (sfull[sel] < tf) & (sbox[sel] >= tb)
        P[use] = Bbox[sel][use]
        return P

    sc_repli = np.zeros(len(noms)); sc_rect = np.zeros(len(noms)); choix = []
    for m in METHODES:
        for k in range(K):
            tr = (meths == m) & (pi != k); te = (meths == m) & (pi == k)
            if not te.any():
                continue
            cand = np.quantile(C[tr], np.linspace(0.02, 0.98, 49))
            t_carte = max(cand, key=lambda t: miou_lot(C[tr] >= t, G[tr]).mean())
            sc_repli[te] = miou_lot(C[te] >= t_carte, G[te])
            tf, tb = max(((a, b) for a in T_FULL for b in T_BOITE),
                         key=lambda q: miou_lot(predire(tr, t_carte, *q), G[tr]).mean())
            sc_rect[te] = miou_lot(predire(te, t_carte, tf, tb), G[te]); choix.append((m, k, tf, tb))
    print(f"repli (carte H1)        : {texte(resume(sc_repli, pi, meths))}", flush=True)
    print(f"rectangle + repli (H3b) : {texte(resume(sc_rect, pi, meths))}", flush=True)
    print("gain :", gain_robuste(sc_repli, sc_rect, pi), flush=True)
    print("seuils choisis (méthode, pli, t_full, t_boite) :", choix, flush=True)
    full = np.array([bool(m["full_watermark"]) for m in meta])
    for m in METHODES:
        for f in (False, True):
            s = (meths == m) & (full == f)
            print(f"  {m:10s} {'full' if f else 'partiel'} : repli {sc_repli[s].mean():.4f} -> rectangle {sc_rect[s].mean():.4f}"
                  f"   (meilleure boîte médiane {np.median(sbox[s]):.3f}, image entière médiane {np.median(sfull[s]):.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--recherche", action="store_true"); ap.add_argument("--evaluer", action="store_true")
    ap.add_argument("--methode", choices=METHODES)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--parts", type=int, default=1); ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--tailles", default=None, help="tailles de boîtes de la grille (défaut : 48..512)")
    ap.add_argument("--top", type=int, default=5, help="boîtes grossières raffinées")
    ap.add_argument("--garder", type=int, default=20, help="boîtes enregistrées par image")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--tache", type=Path, default=off)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--cartes", default=str(off / "output" / "cartes_feats.npz"))
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    if args.recherche:
        recherche(args)
    if args.evaluer:
        evaluer(args)


if __name__ == "__main__":
    main()
