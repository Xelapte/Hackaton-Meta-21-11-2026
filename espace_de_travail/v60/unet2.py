"""v60 / petit U-Net « empileur » : il ne lit PAS le tatouage lui-meme, il combine des cartes de signaux deja
calculees (une case = 4x4 px) et apprend a en faire un masque, sur les 510 images du train (5 plis geles).

Entrees (toutes sans label, calculees pareil pour le train et le test) :
  - l'image reduite a 256 px (formes des objets : les trous et les objets transplantes ont une forme d'objet) ;
  - la methode (3 plans constants) ;
  - les 10 cartes de l'equipe (cartes_feats.npz : correlation des logits par position avec le message connu) ;
  - NOUVEAU : 4 plans « boite » issus de la recherche de rectangle (output/rect) : masque de la boite consensus,
    score de la meilleure boite, precision de bits de l'image entiere (regime full / partiel), carte du meilleur
    score des 20 boites couvrant chaque case ;
  - NOUVEAU (option --corr) : correlation « re-tatouer vrai - faux message » (sig_corr.py), 4 plans + energie.
Perte : BCE + (1 - mIoU doux des DEUX classes, par image) = la metrique officielle rendue derivable
        (le U-Net d'origine n'optimisait que le Dice de la classe 1 : il ne « voyait » pas les trous).
Sorties : v60/unet2_<tag>_oof.npz (probabilites hors pli du train + moyenne des modeles sur le test).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.dont_write_bytecode = True
T = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
W = Path("/p/scratch/training2666/huang25/Epi84/depot-coequipier")
V = Path(__file__).resolve().parent
METHODES = ["pixelseal", "trustmark", "stegastamp"]


def charger_boites(split):
    R = {}
    for f in sorted((T / "output/rect").glob(f"{split}_*.npz")):
        z = np.load(f)
        for n, b, s in zip(z["names"].tolist(), z["boites"], z["full"]):
            R[n] = (b, float(s))
    return R


def boite_consensus(b):
    """Mediane des boites a egalite avec la meilleure (TrustMark : plusieurs boites a 1,00 autour de la vraie)."""
    top = b[b[:, 4] >= b[0, 4] - 1e-6]
    return np.median(top[:, :4], axis=0)


def canaux_boites(b, sfull):
    m = np.zeros((4, 128, 128), np.float32)
    y0, y1, x0, x1 = [int(round(v / 4)) for v in boite_consensus(b)]
    m[0, y0:y1, x0:x1] = 1
    m[1] = (b[0, 4] - 0.5) * 2
    m[2] = (sfull - 0.5) * 2
    for bb in b[::-1]:
        y0, y1, x0, x1 = [int(round(v / 4)) for v in bb[:4]]
        m[3, y0:y1, x0:x1] = np.maximum(m[3, y0:y1, x0:x1], max(float(bb[4]) - 0.5, 0) * 2)
    return m


def corr128(S, bloc):
    k = bloc // 8; g = 64 // k
    s = S.reshape(3, g, k, g, k).sum((2, 4))
    return np.kron(s[0] / np.sqrt(s[1] * s[2] + 1e-9), np.ones((128 // g, 128 // g), np.float32))


def canaux_corr(S):
    c = [corr128(S[0], 16), corr128(S[0], 32), corr128(S[1], 16), corr128(S[1], 32)]
    e = S[0][2].reshape(32, 2, 32, 2).sum((1, 3))
    c.append(np.kron(np.log1p(e), np.ones((4, 4), np.float32)))
    return np.stack(c).astype(np.float32)


def bloc(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class UNet(nn.Module):
    """256 x 256 -> 128 x 128."""
    def __init__(self, cin, w=32):
        super().__init__()
        self.e1 = bloc(cin, w, 2)          # 128
        self.e2 = bloc(w, w * 2, 2)        # 64
        self.e3 = bloc(w * 2, w * 4, 2)    # 32
        self.e4 = bloc(w * 4, w * 8, 2)    # 16
        self.d3 = bloc(w * 8 + w * 4, w * 4)
        self.d2 = bloc(w * 4 + w * 2, w * 2)
        self.d1 = bloc(w * 2 + w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.out(d1)[:, 0]


def diedre(x, k, inverse=False):
    """8 symetries du carre (k = 0..7) sur les 2 dernieres dimensions."""
    r, f = k % 4, k // 4
    if not inverse:
        if f:
            x = x.flip(-1)
        return torch.rot90(x, r, (-2, -1))
    x = torch.rot90(x, -r, (-2, -1))
    return x.flip(-1) if f else x


def perte(lg, y):
    p = torch.sigmoid(lg); q = 1 - p; z = 1 - y; eps = 1.0
    i1 = (p * y).sum((1, 2)); u1 = (p + y - p * y).sum((1, 2))
    i0 = (q * z).sum((1, 2)); u0 = (q + z - q * z).sum((1, 2))
    miou = 0.5 * ((i1 + eps) / (u1 + eps) + (i0 + eps) / (u0 + eps))
    return F.binary_cross_entropy_with_logits(lg, y) + (1 - miou).mean()


def miou_lot(B, G):
    B = np.asarray(B).astype(bool); G = np.asarray(G).astype(bool); s = np.zeros(len(B))
    for b, g in ((B, G), (~B, ~G)):
        inter = (b & g).sum((1, 2)); union = (b | g).sum((1, 2))
        s += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return s / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--corr", type=int, default=0); ap.add_argument("--boites", type=int, default=1)
    ap.add_argument("--ms", type=int, default=0)
    ap.add_argument("--perte", default="miou", choices=["miou", "dice"]); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    mtr = json.loads((T / "release/train/metadata.json").read_text())
    mte = json.loads((T / "release/test/metadata.json").read_text())
    if a.smoke:
        mtr, mte = mtr[:40], mte[:12]
    plis = json.loads((W / "watermark/splits.json").read_text())["plis"]
    z = np.load(T / "release/train/masks_128.npz"); GT = dict(zip(z["names"].tolist(), z["masks"]))
    tous = mtr + mte; ntr = len(mtr)
    noms = [m["image_name"] for m in tous]; meths = np.array([m["method"] for m in tous])
    # ---- cartes de l'equipe
    zc = np.load(T / "output/cartes_feats.npz"); idx = {n: i for i, n in enumerate(zc["names"].tolist())}
    Fz = zc["feats"]; C = [np.stack([Fz[idx[n]] for n in noms]).astype(np.float32)]; del Fz
    # ---- boites
    if a.boites:
        Rtr, Rte = charger_boites("train"), charger_boites("test")
        C.append(np.stack([canaux_boites(*(Rtr[n] if i < ntr else Rte[n])) for i, n in enumerate(noms)]))
    # ---- correlation re-tatouage (v60/corr_<methode>.npz)
    if a.corr:
        S = {}
        for m in METHODES:
            zz = np.load(V / f"corr_{m}.npz")
            for n, s in zip(zz["names"].tolist(), zz["sums"]):
                S[n] = s
        cc = np.stack([canaux_corr(S[n]) for n in noms])
        for m in METHODES:                                   # mise a l'echelle par methode (stats du train, sans label)
            tr = np.where(meths[:ntr] == m)[0]; al = np.where(meths == m)[0]
            sd = cc[tr].std((0, 2, 3), keepdims=True) + 1e-6
            mu = cc[tr].mean((0, 2, 3), keepdims=True); mu[:, :4] = 0      # correlations : on garde le zero physique
            cc[al] = (cc[al] - mu) / sd
        C.append(cc)
    # ---- cartes multi-echelles (v60/ms_<methode>.npz de sig_ms.py) : 10 echelles x 2 statistiques, 64 x 64
    if a.ms:
        mm = np.zeros((len(noms), 20, 64, 64), np.float32); pos = {n: i for i, n in enumerate(noms)}
        for m in ("pixelseal", "trustmark"):
            zz = np.load(V / f"ms_{m}.npz"); nz = zz["names"].tolist()
            mp = zz["maps"].astype(np.float32).reshape(len(nz), -1, 64, 64)
            garde = np.array([n in pos for n in nz]); ids = np.array([pos[n] for n in nz if n in pos]); mp = mp[garde]
            tr = ids < ntr                                       # centrage-reduction par methode (stats du train, sans label)
            mu = mp[tr].mean((0, 2, 3), keepdims=True); sd = mp[tr].std((0, 2, 3), keepdims=True) + 1e-6
            mm[ids, :mp.shape[1]] = (mp - mu) / sd
        C.append(mm.repeat(2, axis=2).repeat(2, axis=3))
    C = np.concatenate(C, 1)
    oh = np.zeros((len(noms), 3, 1, 1), np.float32)
    for i, m in enumerate(meths):
        oh[i, METHODES.index(m)] = 1
    imgs = np.zeros((len(noms), 3, 256, 256), np.float32)
    for i, m in enumerate(tous):
        sp = "train" if i < ntr else "test"
        im = Image.open(T / "release" / sp / "images" / f"{m['image_name']}.png").convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        imgs[i] = (np.asarray(im, np.float32) / 255 - 0.5).transpose(2, 0, 1) * 2
    print(f"donnees : {C.shape} cartes, {imgs.shape} images ({time.time() - t0:.0f} s)", flush=True)
    X = torch.cat([torch.from_numpy(imgs), torch.from_numpy(oh).expand(-1, -1, 256, 256),
                   F.interpolate(torch.from_numpy(C), size=256, mode="nearest")], 1).to(dev).half()
    Y = torch.from_numpy(np.stack([GT[n] for n in noms[:ntr]]).astype(np.float32)).to(dev)
    pk = np.array([plis[n] for n in noms[:ntr]]); cin = X.shape[1]
    print(f"entree : {tuple(X.shape)} sur {dev}", flush=True)

    def predire(model, ii):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for j in range(0, len(ii), 64):
                x = X[ii[j:j + 64]].float(); p = 0
                for k in range(8):
                    p = p + diedre(torch.sigmoid(model(diedre(x, k)).float()), k, inverse=True)
                out.append((p / 8).cpu().numpy())
        model.train()
        return np.concatenate(out)

    oof = np.zeros((ntr, 128, 128), np.float32); pte = np.zeros((len(mte), 128, 128), np.float32)
    ite = np.arange(ntr, len(noms)); nmod = 0
    for k in range(5):
        itr = np.where(pk != k)[0]; iva = np.where(pk == k)[0]
        for s in range(a.seeds):
            torch.manual_seed(1000 * k + s); rng = np.random.default_rng(1000 * k + s)
            model = UNet(cin, a.w).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
            npas = a.epochs * ((len(itr) + a.bs - 1) // a.bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=npas, pct_start=0.1)
            for ep in range(a.epochs):
                perm = rng.permutation(itr)
                for j in range(0, len(perm), a.bs):
                    ii = perm[j:j + a.bs]
                    if len(ii) < 2:
                        sched.step(); continue
                    x = X[ii].float(); y = Y[ii]; kk = int(rng.integers(8))
                    x = diedre(x, kk); y = diedre(y, kk)
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        lg = model(x).float()
                    if a.perte == "miou":
                        loss = perte(lg, y)
                    else:
                        p = torch.sigmoid(lg)
                        dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)
                        loss = F.binary_cross_entropy_with_logits(lg, y) + dice.mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            pv = predire(model, iva); oof[iva] += pv / a.seeds
            pte += predire(model, ite); nmod += 1
            sc = miou_lot(pv >= 0.5, Y[iva].cpu().numpy())
            print(f"pli {k} graine {s} : mIoU val (seuil 0,5) {sc.mean():.4f}  "
                  + "  ".join(f"{m[:5]} {sc[meths[iva] == m].mean():.4f}" for m in METHODES if (meths[iva] == m).any())
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
    pte /= nmod
    sc = miou_lot(oof >= 0.5, Y.cpu().numpy())
    print(f"HORS PLI (seuil 0,5, moyenne des graines) : {sc.mean():.4f}  "
          + "  ".join(f"{m[:5]} {sc[meths[:ntr] == m].mean():.4f}" for m in METHODES), flush=True)
    if not a.smoke:
        np.savez_compressed(V / f"unet2_{a.tag}_oof.npz", names=np.array(noms[:ntr]), methods=meths[:ntr], pli=pk,
                            probs=oof.astype(np.float16), test_names=np.array(noms[ntr:]), test_methods=meths[ntr:],
                            test_probs=pte.astype(np.float16))
        print(f"ecrit unet2_{a.tag}_oof.npz ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
