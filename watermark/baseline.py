"""Tâche 1 (localisation du tatouage) : deux baselines sur les outils officiels (wmloc).

  --mode window : sans entraînement. On connaît la méthode et le message de chaque image :
                  on décode des fenêtres glissantes avec le décodeur de la méthode, on garde
                  le taux de bits justes de chaque fenêtre, on moyenne sur les pixels, puis on
                  seuille. Le seuil (et la taille de fenêtre) est réglé sur le train, par méthode.
                  Enregistre aussi les cartes (window_maps.npz) pour le mode unet.
  --mode unet   : un petit réseau de segmentation (image + passe-haut + méthode [+ carte du
                  mode window]) -> grille 128 x 128, entraîné sur 80 % du train, validé sur 20 %.

Usage (GPU 2, venv de la tâche) :
  bash slurm/soumettre.sh 2 wm_window --venv watermark python watermark/baseline.py --mode window
  bash slurm/soumettre.sh 2 wm_unet   --venv watermark python watermark/baseline.py --mode unet \\
      --maps $OFFICIEL/Hackathon_Watermark_Localisation_Task/output/window_maps.npz
Sortie : output/<mode>.pt au format de soumission ({"image_names", "masks" bool [N,128,128]}),
et le score local (mIoU officiel) dans le journal des scores.
Soumettre : submit(chemin) de submission_template.py, sur le nœud de login (internet).
Test sans GPU ni données : python watermark/baseline.py --fake --mode unet --epochs 1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
METHODES = ["pixelseal", "trustmark", "stegastamp"]


# ----------------------------------------------------------------------------- données
def miou_image(p, g):
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


def score(preds, gts):
    return float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)]))


def charger(args):
    """Liste de dicts (image, image_name, method, message[, mask128]) pour train et test."""
    if args.fake:
        rng = np.random.default_rng(0)
        def fake(n, train):
            out = []
            for i in range(n):
                m = METHODES[i % 3]
                it = {"image": rng.integers(0, 256, (512, 512, 3), dtype=np.uint8),
                      "image_name": f"img{i}", "method": m,
                      "message": rng.integers(0, 2, 256 if m == "pixelseal" else 100).astype(np.uint8)}
                if train:
                    mk = np.zeros((128, 128), np.uint8)
                    y, x = rng.integers(0, 64, 2); mk[y:y + 64, x:x + 64] = 1
                    it["mask128"] = mk
                out.append(it)
            return out
        return fake(24, True), fake(12, False)
    sys.path.insert(0, str(args.tache))
    from wmloc.data import LocalisationDataset
    tr = LocalisationDataset(args.data, "train")
    te = LocalisationDataset(args.data, "test")
    t0 = time.time()
    train = [tr[i] for i in range(len(tr))]
    test = [te[i] for i in range(len(te))]
    print(f"Chargé : {len(train)} train, {len(test)} test ({time.time() - t0:.0f} s)")
    return train, test


def ecrire_soumission(test, masks, chemin):
    names = [it["image_name"] for it in test]
    m = torch.from_numpy(np.stack(masks).astype(bool))
    assert m.shape == (len(names), 128, 128) and len(set(names)) == len(names)
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": names, "masks": m}, chemin)
    print(f"Soumission écrite : {chemin} ({len(names)} masques, {os.path.getsize(chemin) / 1e6:.1f} Mo)")


def journal(config, val):
    try:
        from common.scores import log_score
        log_score("watermark", config, val, notes="mIoU local (validation)")
    except Exception as e:                       # le journal ne doit jamais bloquer un run
        print("journal des scores non écrit :", e)


# ----------------------------------------------------------------------------- mode window
class FauxTatouage:
    def __init__(self, nbits):
        self.nbits = nbits
    def logits(self, x):
        return torch.randn(len(x), self.nbits)


def carte_fenetres(wm, img_u8, msg, tailles, pas, dev, bs=64):
    """Carte 128 x 128 : moyenne, pour chaque case, du taux de bits justes des fenêtres qui la couvrent."""
    x = torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255)[None]
    signe = torch.from_numpy(msg.astype(np.float32) * 2 - 1)
    acc = torch.zeros(512, 512); cnt = torch.zeros(512, 512)
    for t in tailles:
        pos = [(y, xx) for y in range(0, 512 - t + 1, pas) for xx in range(0, 512 - t + 1, pas)]
        for i in range(0, len(pos), bs):
            lot = pos[i:i + bs]
            crops = torch.cat([x[:, :, y:y + t, xx:xx + t] for y, xx in lot]).to(dev)
            lg = wm.logits(crops).float().cpu()
            justes = ((lg * signe[None]) > 0).float().mean(1)          # taux de bits justes
            for (y, xx), a in zip(lot, justes):
                acc[y:y + t, xx:xx + t] += a; cnt[y:y + t, xx:xx + t] += 1
    m = acc / cnt.clamp(min=1)
    return F.avg_pool2d(m[None, None], 4)[0, 0].numpy()                # 128 x 128


def mode_window(args, train, test):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.fake:
        sys.path.insert(0, str(args.tache))
        from wmloc.watermarks import load_watermark
    tailles = [int(t) for t in args.tailles.split(",")]
    cartes, seuils = {}, {}
    for meth in METHODES:
        wm = FauxTatouage(256 if meth == "pixelseal" else 100) if args.fake else \
            load_watermark(meth, models_dir=args.tache / "models", device=dev)
        tr = [it for it in train if it["method"] == meth]
        te = [it for it in test if it["method"] == meth]
        t0 = time.time()
        for it in tr + te:
            cartes[it["image_name"]] = carte_fenetres(wm, it["image"], it["message"], tailles, args.pas, dev)
        # seuil qui maximise le mIoU sur le train de cette méthode
        best = (-1, 0.5)
        for s in np.linspace(0.4, 0.95, 23):
            sc = score([(cartes[it["image_name"]] >= s).astype(np.uint8) for it in tr],
                       [it["mask128"] for it in tr])
            best = max(best, (sc, s))
        seuils[meth] = best[1]
        print(f"{meth}: seuil {best[1]:.3f} -> mIoU train {best[0]:.4f} ({time.time() - t0:.0f} s)")
        del wm; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    preds = [(cartes[it["image_name"]] >= seuils[it["method"]]).astype(np.uint8) for it in train]
    sc = score(preds, [it["mask128"] for it in train])
    print(f"mIoU train (seuils réglés dessus, donc optimiste) : {sc:.4f}")
    out_dir = Path(args.out_dir)
    np.savez_compressed(out_dir / "window_maps.npz", names=np.array(list(cartes)),
                        maps=np.stack(list(cartes.values())).astype(np.float16))
    ecrire_soumission(test, [(cartes[it["image_name"]] >= seuils[it["method"]]).astype(np.uint8)
                             for it in test], out_dir / "window.pt")
    journal({"mode": "window", "tailles": args.tailles, "pas": args.pas,
             "seuils": {k: round(float(v), 3) for k, v in seuils.items()}}, sc)


# ----------------------------------------------------------------------------- mode unet
def bloc(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class PetitUNet(nn.Module):
    """512 x 512 -> 128 x 128. Encodeur 512->256->128->64->32, décodeur jusqu'à 128."""
    def __init__(self, cin, w=32):
        super().__init__()
        self.e1 = bloc(cin, w, 2)        # 256
        self.e2 = bloc(w, w * 2, 2)      # 128
        self.e3 = bloc(w * 2, w * 4, 2)  # 64
        self.e4 = bloc(w * 4, w * 8, 2)  # 32
        self.d3 = bloc(w * 8 + w * 4, w * 4)
        self.d2 = bloc(w * 4 + w * 2, w * 2)
        self.out = nn.Conv2d(w * 2, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        return self.out(d2)[:, 0]        # logits 128 x 128


def entree(items, cartes, dev):
    x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div(255).to(dev)
    passe_haut = x - F.avg_pool2d(x, 5, 1, 2)                          # le tatouage est dans les hautes fréquences
    meth = torch.zeros(len(items), 3, 512, 512, device=dev)
    for i, it in enumerate(items):
        meth[i, METHODES.index(it["method"])] = 1
    chans = [(x - 0.5) / 0.5, passe_haut * 10, meth]
    if cartes is not None:
        c = torch.from_numpy(np.stack([cartes[it["image_name"]] for it in items]).astype(np.float32)).to(dev)
        chans.append(F.interpolate(c[:, None], size=512, mode="nearest"))
    return torch.cat(chans, 1)


def mode_unet(args, train, test):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    cartes = None
    if args.maps:
        z = np.load(args.maps)
        cartes = dict(zip(z["names"].tolist(), z["maps"].astype(np.float32)))
    idx = rng.permutation(len(train)); nval = max(1, int(len(train) * 0.2))
    val = [train[i] for i in idx[:nval]]; tr = [train[i] for i in idx[nval:]]
    if args.toutes:                                   # pour la soumission finale : tout le train
        tr = train
    model = PetitUNet(9 + (cartes is not None), args.largeur).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * ((len(tr) + args.bs - 1) // args.bs))
    use_amp = dev == "cuda"

    def predire(items, seuil=0.5):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for i in range(0, len(items), args.bs):
                p = torch.sigmoid(model(entree(items[i:i + args.bs], cartes, dev)).float())
                out += list(p.cpu().numpy())
        model.train()
        return out

    t0 = time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(len(tr)); tot = 0.0
        for i in range(0, len(tr), args.bs):
            lot = [tr[j] for j in perm[i:i + args.bs]]
            x = entree(lot, cartes, dev)
            y = torch.from_numpy(np.stack([it["mask128"] for it in lot]).astype(np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                lg = model(x).float()
            p = torch.sigmoid(lg)
            dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(lg, y) + dice.mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(lot)
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            pv = predire(val)
            sv = score([(q >= 0.5).astype(np.uint8) for q in pv], [it["mask128"] for it in val])
            print(f"époque {ep:3d}  perte {tot / len(tr):.4f}  mIoU val {sv:.4f}  ({time.time() - t0:.0f} s)")
    pv = predire(val)
    best = max((score([(q >= s).astype(np.uint8) for q in pv], [it["mask128"] for it in val]), s)
               for s in np.linspace(0.2, 0.8, 13))
    print(f"mIoU val final : {best[0]:.4f} (seuil {best[1]:.2f})"
          + ("  [attention : --toutes, la validation est dans l'entraînement]" if args.toutes else ""))
    for meth in METHODES:
        v = [(q, it) for q, it in zip(pv, val) if it["method"] == meth]
        if v:
            print(f"  {meth}: {score([(q >= best[1]).astype(np.uint8) for q, _ in v], [it['mask128'] for _, it in v]):.4f}")
    pt = predire(test)
    out_dir = Path(args.out_dir)
    nom = f"unet{'_maps' if cartes is not None else ''}"
    ecrire_soumission(test, [(q >= best[1]).astype(np.uint8) for q in pt], out_dir / f"{nom}.pt")
    torch.save(model.state_dict(), out_dir / f"{nom}_poids.pt")
    journal({"mode": nom, "epochs": args.epochs, "lr": args.lr, "largeur": args.largeur,
             "seuil": round(float(best[1]), 2), "toutes": args.toutes}, best[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--mode", choices=["window", "unet"], required=True)
    ap.add_argument("--tache", type=Path, default=off, help="dossier de la tâche (wmloc, models)")
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--tailles", default="128,256", help="mode window : tailles des fenêtres")
    ap.add_argument("--pas", type=int, default=64, help="mode window : pas des fenêtres")
    ap.add_argument("--maps", default=None, help="mode unet : window_maps.npz comme entrée en plus")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--largeur", type=int, default=32)
    ap.add_argument("--toutes", action="store_true", help="mode unet : entraîner sur tout le train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    if args.fake:
        args.out_dir = "/tmp/wm_fake"; args.pas = 192; args.tailles = "256"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train, test = charger(args)
    (mode_window if args.mode == "window" else mode_unet)(args, train, test)


if __name__ == "__main__":
    main()
