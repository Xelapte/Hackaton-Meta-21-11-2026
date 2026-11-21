"""Tâche 1 : détecteur appris v2 (à la manière de Watermark Anything), un modèle par méthode.

Entraîné UNIQUEMENT sur les données synthétiques v2 (synth_v2.py) : les 510 vraies images du train ne servent
qu'à mesurer (rapport à chaque époque, sans sélection : nombre d'époques fixé d'avance). Sortie : carte de
probabilité « tatoué par cette méthode » 128 x 128, sans utiliser le message.
Réseau : pleine résolution dans les deux premières couches (le tatouage vit dans les hautes fréquences),
puis encodeur jusqu'à 16 x 16 (contexte) et décodeur jusqu'à 128 x 128. Pas de miroir ni de rotation (H4).
Après l'entraînement : prédiction des images train et test de la méthode -> <out_dir>/det_v2/<méthode>.npz,
utilisée comme repli par regle.py (--repli det).

  bash slurm/soumettre.sh 3 wm_det_tm --venv watermark python watermark/detecteur.py --methode trustmark
  python watermark/detecteur.py --methode trustmark --fake
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
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from watermark.baseline import bloc  # noqa: E402
from watermark.protocole import miou_lot  # noqa: E402


class Detecteur(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(6, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.GELU(),
                                  nn.Conv2d(w, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.GELU())   # 512
        self.d1 = bloc(w, 2 * w, 2)          # 256
        self.d2 = bloc(2 * w, 4 * w, 2)      # 128
        self.d3 = bloc(4 * w, 6 * w, 2)      # 64
        self.d4 = bloc(6 * w, 8 * w, 2)      # 32
        self.d5 = bloc(8 * w, 8 * w, 2)      # 16
        self.u4 = bloc(16 * w, 6 * w)
        self.u3 = bloc(12 * w, 4 * w)
        self.u2 = bloc(8 * w, 3 * w)
        self.out = nn.Conv2d(3 * w, 1, 1)

    def forward(self, x):
        x = torch.cat([x * 2 - 1, (x - F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)) * 20], 1)
        s = self.stem(x); e1 = self.d1(s); e2 = self.d2(e1); e3 = self.d3(e2); e4 = self.d4(e3); e5 = self.d5(e4)
        u4 = self.u4(torch.cat([F.interpolate(e5, scale_factor=2), e4], 1))
        u3 = self.u3(torch.cat([F.interpolate(u4, scale_factor=2), e3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, scale_factor=2), e2], 1))
        return self.out(u2)[:, 0]


class Synth(torch.utils.data.Dataset):
    def __init__(self, pre):
        self.im = np.load(f"{pre}_images.npy", mmap_mode="r"); self.m = np.load(f"{pre}_masks.npy")
    def __len__(self):
        return len(self.m)
    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(self.im[i])), torch.from_numpy(self.m[i].astype(np.float32))


@torch.no_grad()
def predire(model, imgs, dev, bs=16):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).to(dev).permute(0, 3, 1, 2).float().div(255)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--methode", choices=["pixelseal", "trustmark", "stegastamp"], required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--largeur", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    if args.fake:
        rng = np.random.default_rng(0); out = Path("/tmp/wm_fake_det"); (out / "synth_v2").mkdir(parents=True, exist_ok=True)
        pre = out / "synth_v2" / args.methode
        np.save(f"{pre}_images.npy", rng.integers(0, 255, (8, 512, 512, 3), dtype=np.uint8))
        np.save(f"{pre}_masks.npy", (rng.random((8, 128, 128)) < .5).astype(np.uint8))
        reels = {s: [(f"{s}{i}", rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
                      (rng.random((128, 128)) < .5) if s == "train" else None) for i in range(4)] for s in ("train", "test")}
        args.epochs, args.bs, args.workers, args.largeur = 1, 4, 0, 8
    else:
        pre = out / "synth_v2" / args.methode
        reels = {}
        for s in ("train", "test"):
            meta = [m for m in json.loads((args.data / s / "metadata.json").read_text()) if m["method"] == args.methode]
            M = None
            if s == "train":
                z = np.load(args.data / "train" / "masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
            reels[s] = [(m["image_name"], np.asarray(Image.open(args.data / s / "images" / f"{m['image_name']}.png").convert("RGB")),
                         None if M is None else M[m["image_name"]].astype(bool)) for m in meta]
    ds = Synth(pre)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
                                     pin_memory=dev == "cuda", persistent_workers=args.workers > 0)
    model = Detecteur(args.largeur).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    print(f"{args.methode} : {len(ds)} exemples synthétiques, {len(reels['train'])} vraies images de train (mesure seule), "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M paramètres", flush=True)
    imgs_tr = [x[1] for x in reels["train"]]; G = np.stack([x[2] for x in reels["train"]])
    t0 = time.time()
    for ep in range(args.epochs):
        tot = 0.0
        for k, (x, y) in enumerate(dl):
            x = x.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(255); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lg = model(x).float()
            p = torch.sigmoid(lg)
            dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(lg, y) + dice.mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
            if k % 200 == 0:
                print(f"  époque {ep} lot {k}/{len(dl)} perte {tot / (k + 1):.4f} ({time.time() - t0:.0f} s)", flush=True)
        P = predire(model, imgs_tr, dev)
        print(f"époque {ep} : perte {tot / len(dl):.4f} ; vraies images du train (mesure, pas de sélection) : "
              f"mIoU au seuil 0,5 {miou_lot(P >= 0.5, G).mean():.4f} ({time.time() - t0:.0f} s)", flush=True)
    (out / "det_v2").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "det_v2" / f"{args.methode}_poids.pt")
    Ptr = predire(model, imgs_tr, dev); Pte = predire(model, [x[1] for x in reels["test"]], dev)
    np.savez_compressed(out / "det_v2" / f"{args.methode}.npz",
                        names=np.array([x[0] for x in reels["train"]] + [x[0] for x in reels["test"]]),
                        probs=np.concatenate([Ptr, Pte]).astype(np.float16))
    print(f"Écrit : {out / 'det_v2' / (args.methode + '.npz')} ; mIoU final (train réel, seuil 0,5) "
          f"{miou_lot(Ptr >= 0.5, G).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
