"""Tâche 1 (localisation du tatouage) : deux baselines sur les outils officiels (wmloc).

  --mode window : sans entraînement. On connaît la méthode et le message de chaque image :
                  on décode des fenêtres glissantes avec le décodeur de la méthode, on garde
                  le taux de bits justes de chaque fenêtre, on moyenne sur les pixels, puis on
                  seuille. Le seuil (et la taille de fenêtre) est réglé sur le train, par méthode.
                  Enregistre aussi les cartes (window_maps.npz) pour le mode unet.
  --mode direct : sans entraînement, sans fenêtres. Corrélation au message des logits par position
                  (PixelSeal, TrustMark), résolution d'entrée choisie sur le train ; StegaStamp :
                  re-tatouer et corréler (--ss reembed, défaut) ou carte window (--ss window --maps). Écrit output/direct.pt et direct_maps.npz.
  --mode cartes : pile de cartes par image (PixelSeal/TrustMark par position, StegaStamp re-tatouage),
                  écrite dans output/cartes_feats.npz, pour --mode unet --feats.
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


# ----------------------------------------------------------------------------- mode direct
@torch.no_grad()
def carte_directe(wm, meth, img_u8, msg, res, dev):
    """Corrélation au message des logits par position (sans fenêtres), à la résolution d'entrée res.
    PixelSeal : ConvNeXt + tête linéaire ; TrustMark : ResNet-50 layer4 + fc. Renvoie une carte 128 x 128."""
    x = torch.from_numpy(img_u8).permute(2, 0, 1)[None].float().div(255).to(dev)
    signe = torch.from_numpy(msg.astype(np.float32) * 2 - 1).to(dev)
    if res != 512:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    if meth == "pixelseal":
        d = wm.det
        emb = d.pixel_decoder.output_upscaling(d.convnext(d.preprocess(x)))
        lin = d.pixel_decoder.linear
        pp = (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
    else:
        n = wm.net; h = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
        for k in ("layer1", "layer2", "layer3", "layer4"):
            h = getattr(n, k)(h)
        pp = torch.einsum("bchw,oc->bohw", h, n.fc.weight) + n.fc.bias[None, :, None, None]
    c = (pp[0].float() * signe[:, None, None]).mean(0)
    return F.interpolate(c[None, None], size=128, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def passe_haut(x, k):
    return x - F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)


def carte_reembed(img_u8, re_u8, bloc, dev, k=5):
    """Re-tatouer et corréler : corrélation normalisée locale (blocs de `bloc` pixels) entre passe-haut(I) et
    passe-haut(R'), R' = embed(I, message) - I. Renvoie une carte 128 x 128."""
    i = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float().to(dev)
    r = torch.from_numpy(np.ascontiguousarray(re_u8)).permute(2, 0, 1)[None].float().to(dev) - i
    a = passe_haut(i, k); b = passe_haut(r, k)
    pool = lambda t: F.avg_pool2d(t.sum(1, keepdim=True), bloc, 1, bloc // 2, count_include_pad=False)
    c = pool(a * b) / (pool(a * a).sqrt() * pool(b * b).sqrt() + 1e-6)
    return F.avg_pool2d(c[..., :512, :512], 4)[0, 0].cpu().numpy()


def seuil_optimal(cartes, gts, n=41):
    v = np.concatenate([c.ravel() for c in cartes])
    return max((score([(c >= s).astype(np.uint8) for c in cartes], gts), float(s))
               for s in np.quantile(v, np.linspace(0.02, 0.98, n)))


def mode_direct(args, train, test):
    """PixelSeal et TrustMark : carte de corrélation par position, meilleure résolution choisie sur le train.
    StegaStamp (pas de sortie par position) : carte du mode window (--maps)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(args.tache))
    from wmloc.watermarks import load_watermark
    fen = None
    if args.maps:
        z = np.load(args.maps)
        fen = dict(zip(z["names"].tolist(), z["maps"].astype(np.float32)))
    resolutions = {"pixelseal": [int(r) for r in args.res_ps.split(",")],
                   "trustmark": [int(r) for r in args.res_tm.split(",")]}
    choix, seuils, toutes_cartes = {}, {}, {}
    for meth in METHODES:
        tr = [it for it in train if it["method"] == meth]
        te = [it for it in test if it["method"] == meth]
        gts = [it["mask128"] for it in tr]
        if meth == "stegastamp" and args.ss == "window":
            if fen is None:
                print("stegastamp : pas de --maps, prédiction tout à 0"); choix[meth] = "zeros"
                cartes = {it["image_name"]: np.zeros((128, 128), np.float32) for it in tr + te}
                seuils[meth] = 0.5
            else:
                cartes = {it["image_name"]: fen[it["image_name"]] for it in tr + te}
                sc, seuils[meth] = seuil_optimal([cartes[it["image_name"]] for it in tr], gts)
                choix[meth] = "window"; print(f"stegastamp : window -> mIoU train {sc:.4f}")
            toutes_cartes.update(cartes); continue
        if meth == "stegastamp":                      # re-tatouer et corréler, taille de bloc choisie sur le train
            wm = load_watermark(meth, models_dir=args.tache / "models", device=dev)
            re = {it["image_name"]: wm.embed(it["image"], it["message"]) for it in tr + te}
            best = None
            for b in [int(v) for v in args.blocs_ss.split(",")]:
                cartes = {it["image_name"]: carte_reembed(it["image"], re[it["image_name"]], b, dev) for it in tr + te}
                sc, s_ = seuil_optimal([cartes[it["image_name"]] for it in tr], gts)
                print(f"stegastamp : re-tatouage bloc {b} -> mIoU train {sc:.4f} (seuil {s_:+.3f})")
                if best is None or sc > best[0]:
                    best = (sc, s_, f"reembed_{b}", cartes)
            _, seuils[meth], choix[meth], cartes = best
            toutes_cartes.update(cartes); del wm; continue
        wm = load_watermark(meth, models_dir=args.tache / "models", device=dev)
        best = None
        for r in resolutions[meth]:
            cartes = {it["image_name"]: carte_directe(wm, meth, it["image"], it["message"], r, dev) for it in tr + te}
            sc, s = seuil_optimal([cartes[it["image_name"]] for it in tr], gts)
            print(f"{meth} : résolution {r} -> mIoU train {sc:.4f} (seuil {s:+.3f})")
            if best is None or sc > best[0]:
                best = (sc, s, r, cartes)
        _, seuils[meth], choix[meth], cartes = best
        toutes_cartes.update(cartes)
        del wm; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    preds = [(toutes_cartes[it["image_name"]] >= seuils[it["method"]]).astype(np.uint8) for it in train]
    sc = score(preds, [it["mask128"] for it in train])
    for meth in METHODES:
        v = [(p, it) for p, it in zip(preds, train) if it["method"] == meth]
        print(f"  {meth}: {score([p for p, _ in v], [it['mask128'] for _, it in v]):.4f} ({choix[meth]})")
    print(f"mIoU train (un seuil par méthode réglé dessus) : {sc:.4f}")
    out_dir = Path(args.out_dir)
    np.savez_compressed(out_dir / "direct_maps.npz", names=np.array(list(toutes_cartes)),
                        maps=np.stack(list(toutes_cartes.values())).astype(np.float16))
    ecrire_soumission(test, [(toutes_cartes[it["image_name"]] >= seuils[it["method"]]).astype(np.uint8)
                             for it in test], out_dir / "direct.pt")
    journal({"mode": "direct", "resolutions": {k: v for k, v in choix.items()}, "ss": args.ss,
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


# ----------------------------------------------------------------------------- mode cartes
def noms_canaux(args):
    return ([f"ps_{r}" for r in args.feat_ps.split(",")] + [f"tm_{r}" for r in args.feat_tm.split(",")]
            + [f"ss_{b}" for b in args.feat_ss.split(",")])


def cartes_image(wm, meth, img_u8, msg, args, dev, re_u8=None):
    """Les canaux bruts (non normalisés) d'une image : (indice du premier canal de la méthode, liste de cartes)."""
    rps = [int(r) for r in args.feat_ps.split(",")]; rtm = [int(r) for r in args.feat_tm.split(",")]
    if meth == "stegastamp":
        re = wm.embed(img_u8, msg) if re_u8 is None else re_u8
        return len(rps) + len(rtm), [carte_reembed(img_u8, re, int(b), dev) for b in args.feat_ss.split(",")]
    rs = rps if meth == "pixelseal" else rtm
    return (0 if meth == "pixelseal" else len(rps)), [carte_directe(wm, meth, img_u8, msg, r, dev) for r in rs]


def mode_cartes(args, train, test):
    """Pile de cartes 128 x 128 par image (train et test), entrée du U-Net (--feats) :
    PixelSeal (corrélation, résolutions --feat_ps), TrustMark (--feat_tm), StegaStamp (re-tatouage, blocs
    --feat_ss). Les canaux d'une autre méthode valent 0. Chaque canal est centré-réduit avec les statistiques
    du train de sa méthode ; mu et sd sont enregistrés (réutilisés par watermark/synthetique.py)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(args.tache))
    from wmloc.watermarks import load_watermark
    noms = noms_canaux(args); tous = train + test
    feats = np.zeros((len(tous), len(noms), 128, 128), np.float32)
    mu = np.zeros(len(noms), np.float32); sd = np.ones(len(noms), np.float32)
    for meth in METHODES:
        t0 = time.time()
        wm = load_watermark(meth, models_dir=args.tache / "models", device=dev)
        idx = [i for i, it in enumerate(tous) if it["method"] == meth]
        for i in idx:
            d, cs = cartes_image(wm, meth, tous[i]["image"], tous[i]["message"], args, dev)
            feats[i, d:d + len(cs)] = np.stack(cs)
        itr = [i for i in idx if i < len(train)]
        for j in range(d, d + len(cs)):                   # normalisation par méthode (stats du train)
            mu[j], sd[j] = feats[itr, j].mean(), feats[itr, j].std() + 1e-6
            feats[idx, j] = (feats[idx, j] - mu[j]) / sd[j]
        print(f"{meth} : {len(idx)} images ({time.time() - t0:.0f} s)")
        del wm; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    chemin = Path(args.out_dir) / "cartes_feats.npz"
    np.savez(chemin, names=np.array([it["image_name"] for it in tous]), canaux=np.array(noms),
             feats=feats.astype(np.float16), mu=mu, sd=sd)
    print(f"Cartes : {chemin} ({feats.shape}, canaux {noms})")


# ----------------------------------------------------------------------------- mode unet (suite)
def entree(items, cartes, dev, feats=None):
    x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div(255).to(dev)
    passe_haut = x - F.avg_pool2d(x, 5, 1, 2)                          # le tatouage est dans les hautes fréquences
    meth = torch.zeros(len(items), 3, 512, 512, device=dev)
    for i, it in enumerate(items):
        meth[i, METHODES.index(it["method"])] = 1
    chans = [(x - 0.5) / 0.5, passe_haut * 10, meth]
    if cartes is not None:
        c = torch.from_numpy(np.stack([cartes[it["image_name"]] for it in items]).astype(np.float32)).to(dev)
        chans.append(F.interpolate(c[:, None], size=512, mode="nearest"))
    if feats is not None:
        f = torch.from_numpy(np.stack([feats[it["image_name"]] for it in items]).astype(np.float32)).to(dev)
        chans.append(F.interpolate(f, size=512, mode="nearest"))
    return torch.cat(chans, 1)


def augmenter(x, y, rng):
    """Retournements et rotations de 90° appliqués ensemble à l'entrée (B x C x 512 x 512) et au masque."""
    k = int(rng.integers(4))
    if k:
        x = torch.rot90(x, k, (2, 3)); y = torch.rot90(y, k, (1, 2))
    if rng.random() < 0.5:
        x = x.flip(3); y = y.flip(2)
    return x, y


def charger_synth(dossier, feats):
    """Exemples de watermark/synthetique.py comme items (image et cartes lues à la demande, en memmap)."""
    items = []
    for meth in METHODES:
        pre = Path(dossier) / meth
        if not Path(f"{pre}_masks.npy").exists():
            print(f"synthétique : rien pour {meth}"); continue
        ims = np.load(f"{pre}_images.npy", mmap_mode="r"); ms = np.load(f"{pre}_masks.npy")
        fs = np.load(f"{pre}_feats.npy", mmap_mode="r")
        for i in range(len(ms)):
            nom = f"synth_{meth}_{i:05d}"
            feats[nom] = fs[i]
            items.append({"image": ims[i], "image_name": nom, "method": meth, "mask128": ms[i]})
        print(f"synthétique : {len(ms)} exemples {meth}, part tatouée moyenne {ms.mean():.3f}")
    return items


def entrainer(args, tr, val, cartes, feats, dev, rng, cin, synth=()):
    """Entraîne un PetitUNet sur tr (+ --synth_par_epoque exemples synthétiques tirés à chaque époque) ;
    renvoie une fonction de prédiction (probabilités 128 x 128)."""
    model = PetitUNet(cin, args.largeur).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_synth = min(args.synth_par_epoque, len(synth)) if synth else 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs * ((len(tr) + n_synth + args.bs - 1) // args.bs))
    use_amp = dev == "cuda"

    def predire(items):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for i in range(0, len(items), args.bs):
                p = torch.sigmoid(model(entree(items[i:i + args.bs], cartes, dev, feats)).float())
                out += list(p.cpu().numpy())
        model.train()
        return out

    t0 = time.time()
    for ep in range(args.epochs):
        ep_items = list(tr) + ([synth[j] for j in rng.choice(len(synth), n_synth, replace=False)] if n_synth else [])
        perm = rng.permutation(len(ep_items)); tot = 0.0
        for i in range(0, len(ep_items), args.bs):
            lot = [ep_items[j] for j in perm[i:i + args.bs]]
            x = entree(lot, cartes, dev, feats)
            y = torch.from_numpy(np.stack([it["mask128"] for it in lot]).astype(np.float32)).to(dev)
            if args.aug:
                x, y = augmenter(x, y, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                lg = model(x).float()
            p = torch.sigmoid(lg)
            dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(lg, y) + dice.mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(lot)
        tot = tot * len(tr) / len(ep_items)
        if val and (ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1):
            pv = predire(val)
            sv = score([(q >= 0.5).astype(np.uint8) for q in pv], [it["mask128"] for it in val])
            print(f"  époque {ep:3d}  perte {tot / len(tr):.4f}  mIoU val {sv:.4f}  ({time.time() - t0:.0f} s)")
    return predire, model


def mode_unet(args, train, test):
    """--folds 1 : comme avant (80 % / 20 %). --folds K > 1 : validation croisée en K plis, score hors pli
    sur tout le train, et prédiction du test = moyenne des K modèles."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    cartes = feats = None
    if args.maps:
        z = np.load(args.maps)
        cartes = dict(zip(z["names"].tolist(), z["maps"].astype(np.float32)))
    if args.feats:
        z = np.load(args.feats)
        feats = dict(zip(z["names"].tolist(), z["feats"]))
        print(f"Cartes en entrée : {z['canaux'].tolist()}")
    synth = charger_synth(args.synth, feats) if args.synth else []
    cin = 9 + (cartes is not None) + (len(next(iter(feats.values()))) if feats else 0)
    idx = rng.permutation(len(train))
    if args.plis_geles:                               # protocole : plis de watermark/splits.json
        from watermark.protocole import charger_plis
        pg = charger_plis()
        plis = [np.array([i for i, it in enumerate(train) if pg[it["image_name"]] == k]) for k in range(5)]
    elif args.folds > 1:
        plis = np.array_split(idx, args.folds)
    else:
        nval = max(1, int(len(train) * 0.2)); plis = [idx[:nval]]
    pv_tout, gt_tout, it_tout, pt_somme = [], [], [], None
    for k, pli in enumerate(plis):
        val = [train[i] for i in pli]
        tr = train if args.toutes else [train[i] for i in idx if i not in set(pli.tolist())]
        print(f"Pli {k + 1}/{len(plis)} : {len(tr)} entraînement, {len(val)} validation")
        predire, model = entrainer(args, tr, val, cartes, feats, dev, rng, cin, synth)
        pv_tout += predire(val); it_tout += val
        pt = np.stack(predire(test))
        pt_somme = pt if pt_somme is None else pt_somme + pt
    pt_moy = pt_somme / len(plis)
    gts = [it["mask128"] for it in it_tout]
    if args.plis_geles:                               # score imbriqué : seuil par méthode réglé sur les 4 autres plis
        from watermark.protocole import miou_lot, resume, texte
        P = np.stack(pv_tout); G = np.stack(gts).astype(bool)
        mt = np.array([it["method"] for it in it_tout])
        pk = np.concatenate([np.full(len(pl), k) for k, pl in enumerate(plis)])
        sc = np.zeros(len(P))
        for m in METHODES:
            for k in range(len(plis)):
                tr = (mt == m) & (pk != k); te = (mt == m) & (pk == k)
                t = max(np.linspace(0.2, 0.8, 13), key=lambda t: miou_lot(P[tr] >= t, G[tr]).mean())
                sc[te] = miou_lot(P[te] >= t, G[te])
        r = resume(sc, pk, mt)
        print(f"PROTOCOLE (plis gelés, seuil imbriqué) : {texte(r)}", flush=True)
        np.save(Path(args.out_dir) / f"unet_scores_imbriques_{args.tag}.npy", np.stack([sc, pk]))
    best = max((score([(q >= s).astype(np.uint8) for q in pv_tout], gts), s) for s in np.linspace(0.2, 0.8, 13))
    print(f"mIoU {'hors pli' if args.folds > 1 else 'val'} final : {best[0]:.4f} (seuil {best[1]:.2f}, "
          f"{len(it_tout)} images)" + ("  [attention : --toutes]" if args.toutes else ""))
    for meth in METHODES:
        v = [(q, it) for q, it in zip(pv_tout, it_tout) if it["method"] == meth]
        if v:
            print(f"  {meth}: {score([(q >= best[1]).astype(np.uint8) for q, _ in v], [it['mask128'] for _, it in v]):.4f}")
    out_dir = Path(args.out_dir)
    nom = "unet" + (f"_{args.tag}" if args.tag else "") + ("_maps" if cartes is not None else "") + ("_feats" if feats is not None else "") \
        + ("_synth" if synth else "") + (f"_cv{args.folds}" if args.folds > 1 else "")
    ecrire_soumission(test, [(q >= best[1]).astype(np.uint8) for q in pt_moy], out_dir / f"{nom}.pt")
    np.save(out_dir / f"{nom}_probas_test.npy", pt_moy.astype(np.float16))
    # probabilités hors pli (validation) et du test, pour watermark/postproc.py
    np.savez_compressed(out_dir / f"{nom}_oof.npz",
                        names=np.array([it["image_name"] for it in it_tout]),
                        methods=np.array([it["method"] for it in it_tout]),
                        pli=np.concatenate([np.full(len(pl), k) for k, pl in enumerate(plis)]),
                        probs=np.stack(pv_tout).astype(np.float16),
                        test_names=np.array([it["image_name"] for it in test]),
                        test_methods=np.array([it["method"] for it in test]), test_probs=pt_moy.astype(np.float16))
    torch.save(model.state_dict(), out_dir / f"{nom}_poids.pt")
    journal({"mode": nom, "epochs": args.epochs, "lr": args.lr, "largeur": args.largeur, "folds": args.folds,
             "aug": args.aug, "feats": bool(feats), "seuil": round(float(best[1]), 2), "toutes": args.toutes,
             "seed": args.seed, "plis_geles": args.plis_geles, "tag": args.tag, "synth": len(synth), "synth_par_epoque": args.synth_par_epoque if synth else 0},
            r["moyenne"] if args.plis_geles else best[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    off = Path(os.environ.get("OFFICIEL", ".")) / "Hackathon_Watermark_Localisation_Task"
    ap.add_argument("--mode", choices=["window", "direct", "cartes", "unet"], required=True)
    ap.add_argument("--tache", type=Path, default=off, help="dossier de la tâche (wmloc, models)")
    ap.add_argument("--data", type=Path, default=off / "release")
    ap.add_argument("--out_dir", default=str(off / "output"))
    ap.add_argument("--tailles", default="128,256", help="mode window : tailles des fenêtres")
    ap.add_argument("--pas", type=int, default=64, help="mode window : pas des fenêtres")
    ap.add_argument("--maps", default=None, help="modes unet et direct : window_maps.npz (entrée en plus / StegaStamp)")
    ap.add_argument("--res_ps", default="224,256,320,384,512", help="mode direct : résolutions essayées, PixelSeal")
    ap.add_argument("--ss", choices=["reembed", "window"], default="reembed", help="mode direct : carte StegaStamp")
    ap.add_argument("--blocs_ss", default="8,16,24", help="mode direct : tailles de bloc essayées (re-tatouage)")
    ap.add_argument("--res_tm", default="192,224,245,288,320", help="mode direct : résolutions essayées, TrustMark")
    ap.add_argument("--feat_ps", default="256,384,512", help="mode cartes : résolutions PixelSeal")
    ap.add_argument("--feat_tm", default="224,245,288", help="mode cartes : résolutions TrustMark")
    ap.add_argument("--feat_ss", default="8,16,32,64", help="mode cartes : blocs du re-tatouage StegaStamp")
    ap.add_argument("--feats", default=None, help="mode unet : cartes_feats.npz (mode cartes) en entrée")
    ap.add_argument("--folds", type=int, default=1, help="mode unet : plis de validation croisée (1 = 80/20)")
    ap.add_argument("--synth", default=None, help="mode unet : dossier des exemples synthétiques (synthetique.py)")
    ap.add_argument("--synth_par_epoque", type=int, default=1500, help="mode unet : exemples synthétiques par époque")
    ap.add_argument("--plis_geles", action="store_true", help="mode unet : plis de watermark/splits.json (protocole)")
    ap.add_argument("--tag", default="", help="mode unet : suffixe des fichiers de sortie")
    ap.add_argument("--aug", action="store_true", help="mode unet : retournements et rotations de 90°")
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
    {"window": mode_window, "direct": mode_direct, "cartes": mode_cartes, "unet": mode_unet}[args.mode](args, train, test)


if __name__ == "__main__":
    main()
