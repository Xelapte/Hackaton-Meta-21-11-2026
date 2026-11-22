#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v61 - Localisation de watermark : segmentation APPRISE sur des donnees SYNTHETIQUES.

Idee (a la "Watermark Anything") : on sait fabriquer nous-memes des exemples avec la verite terrain.
  1. On prend une image de fond SANS watermark de la methode etudiee (= les images des deux AUTRES methodes).
  2. On choisit un rectangle (tire des vrais masques d'entrainement des autres methodes : memes formes/tailles
     que les vraies editions), on y incruste un watermark avec l'embedder officiel et un message aleatoire.
  3. On ne garde le watermark que dans un masque (rectangle entier = outpainting, rectangle troue = objet
     retire/ajoute, tache = objet transplante). Le masque est la verite terrain, gratuite.
  4. Un reseau de segmentation (encodeur = l'extracteur officiel de la methode, deja sensible au watermark,
     + petit decodeur type FPN) apprend a predire le masque 128 x 128.
Validation : UNIQUEMENT sur les vraies images d'entrainement de la methode (jamais vues en phase A).
Phase B : affinage sur les vraies images en validation croisee 5 plis (predictions hors-pli = score honnete),
puis prediction du test par la moyenne des 5 modeles.
Ecrit uniquement dans unlearning/v61/.
"""
import argparse, copy, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

TASK = Path("/p/scratch/training2666/huang25/Epi84/Hackathon_Watermark_Localisation_Task")
OUT = Path("/p/scratch/training2666/huang25/Epi84/unlearning/v61")
sys.path.insert(0, str(TASK))
from wmloc.utils import miou_image            # metrique officielle  # noqa: E402
from wmloc.watermarks import load_watermark   # embedders officiels  # noqa: E402

DEV = "cuda"
S = 512
AMP = {"on": True}      # precision mixte bf16 ; coupee automatiquement si un essai echoue


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- donnees
def _png(p):
    return np.asarray(Image.open(p).convert("RGB"))


def _msk(p):
    return np.asarray(Image.open(p).convert("L")) > 127


def load_data():
    tr = json.loads((TASK / "release/train/metadata.json").read_text())
    te = json.loads((TASK / "release/test/metadata.json").read_text())
    with ThreadPoolExecutor(16) as ex:
        tr_img = np.stack(list(ex.map(_png, [TASK / f"release/train/images/{m['image_name']}.png" for m in tr])))
        te_img = np.stack(list(ex.map(_png, [TASK / f"release/test/images/{m['image_name']}.png" for m in te])))
        tr_msk = np.stack(list(ex.map(_msk, [TASK / f"release/train/masks/{m['image_name']}.png" for m in tr])))
    z = np.load(TASK / "release/train/masks_128.npz")
    g = dict(zip(z["names"].tolist(), z["masks"]))
    tr_m128 = np.stack([g[m["image_name"]] for m in tr]).astype(np.uint8)
    return tr, te, tr_img, te_img, tr_msk, tr_m128


# ----------------------------------------------------------------------------- rectangles synthetiques
def _fit(c0, c1, size):
    """Intervalle de longueur `size` centre comme [c0, c1], recale dans [0, 512]."""
    size = int(min(S, max(1, round(size))))
    a = int(round((c0 + c1) / 2 - size / 2))
    a = max(0, min(S - size, a))
    return a, a + size


def aug_mask(m, full, rng):
    """Augmentation geometrique d'un vrai masque : symetries, transposition et (une fois sur deux, pour les
    patchs) translation. Seule la FORME change ; l'incrustation reste faite par le vrai embedder."""
    if rng.random() < 0.5:
        m = m[:, ::-1]
    if rng.random() < 0.5:
        m = m[::-1]
    if rng.random() < 0.5:
        m = m.T
    if not full and rng.random() < 0.5:
        ys, xs = np.where(m)
        if len(ys):
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            dy, dx = int(rng.integers(-y0, S - y1 + 1)), int(rng.integers(-x0, S - x1 + 1))
            out = np.zeros_like(m)
            out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = m[y0:y1, x0:x1]
            m = out
    return np.ascontiguousarray(m)


def rect_from_mask(m, full, rng, max_ar, min_side=8):
    """Rectangle d'incrustation deduit d'un vrai masque. Renvoie (y0, y1, x0, x1), type.
    type : 0 = image entiere watermarkee (avec trous), 1 = patch rectangulaire, 2 = forme d'objet (transplantation)."""
    if full:
        return (0, S, 0, S), 0
    ys, xs = np.where(m)
    if len(ys) == 0:
        return random_rect(rng, max_ar), 1
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = float(y1 - y0), float(x1 - x0)
    kind = 1 if len(ys) / (h * w) >= 0.95 else 2
    if kind == 2:                       # objet transplante : le rectangle source est inconnu -> on varie
        u = rng.random()
        if u < 0.4:
            return (0, S, 0, S), 2      # objet pris dans une image entierement watermarkee
        if u < 0.65:
            f = rng.uniform(1.0, 2.5)
            h, w = h * f, w * f
    if max(h, w) / max(1.0, min(h, w)) > max_ar:    # respecte le rapport d'aspect maximal de la methode
        if h > w:
            w = h / max_ar
        else:
            h = w / max_ar
    h, w = max(h, min_side), max(w, min_side)
    y0, y1 = _fit(y0, y1, h)
    x0, x1 = _fit(x0, x1, w)
    return (y0, y1, x0, x1), kind


def random_rect(rng, max_ar, min_side=16):
    for _ in range(100):
        h, w = int(rng.integers(min_side, S + 1)), int(rng.integers(min_side, S + 1))
        if max(h, w) / min(h, w) <= max_ar:
            break
    y0, x0 = int(rng.integers(0, S - h + 1)), int(rng.integers(0, S - w + 1))
    return (y0, y0 + h, x0, x0 + w)


class Pool:
    """Reserve d'images synthetiques stockee sur le GPU. Un element = une image de fond dont un rectangle a ete
    watermarke par l'embedder officiel (message aleatoire) + un masque (vrai masque augmente, limite au rectangle).
    A chaque batch : image = watermarkee dans le masque, propre ailleurs ; la verite terrain est le masque."""
    P_FOREIGN = (0.10, 0.15, 0.60)      # proba que le contenu HORS masque vienne d'une autre image, selon le type

    def __init__(self, wm, base_np, masks_np, fulls, max_ar, cap, rng, dev=None):
        self.dev = dev or DEV
        self.wm, self.base_np, self.masks_np, self.fulls = wm, base_np, masks_np, fulls
        self.max_ar, self.cap, self.rng = max_ar, cap, rng
        self.base = torch.from_numpy(base_np).to(self.dev).permute(0, 3, 1, 2).contiguous()      # uint8
        self.img = torch.zeros((cap, 3, S, S), dtype=torch.uint8, device=self.dev)
        self.msk = torch.zeros((cap, S, S), dtype=torch.bool, device=self.dev)
        self.bi = torch.zeros(cap, dtype=torch.long, device=self.dev)
        self.kind = torch.zeros(cap, dtype=torch.long, device=self.dev)
        self.rect = torch.zeros((cap, 4), dtype=torch.long, device=self.dev)
        self.n = 0
        self.t_embed = []
        yy, xx = torch.meshgrid(torch.arange(S, device=self.dev), torch.arange(S, device=self.dev), indexing="ij")
        self.yy, self.xx = yy[None], xx[None]
        self.pf = torch.tensor(self.P_FOREIGN, device=self.dev)

    def add(self):
        rng = self.rng
        b = int(rng.integers(len(self.base_np)))
        if rng.random() < 0.9:          # forme tiree d'un vrai masque (d'une AUTRE methode), augmentee
            k = int(rng.integers(len(self.masks_np)))
            m = aug_mask(self.masks_np[k], self.fulls[k], rng)
            rect, kind = rect_from_mask(m, self.fulls[k], rng, self.max_ar)
        else:                           # rectangle aleatoire
            m, rect, kind = None, random_rect(rng, self.max_ar), 1
        y0, y1, x0, x1 = rect
        rm = np.zeros((S, S), bool)
        rm[y0:y1, x0:x1] = True
        m = rm if m is None else (m & rm)
        if kind == 1:                   # patch rectangulaire : le masque est exactement le rectangle incruste
            m = rm
        crop = np.ascontiguousarray(self.base_np[b, y0:y1, x0:x1])
        t = time.time()
        w = np.asarray(self.wm.embed(crop, rng.integers(0, 2, self.wm.nbits)))
        self.t_embed.append(time.time() - t)
        if w.shape != crop.shape:       # securite : taille de sortie differente -> on redimensionne
            w = np.asarray(Image.fromarray(w).resize((crop.shape[1], crop.shape[0]), Image.Resampling.BICUBIC))
        slot = self.n if self.n < self.cap else int(rng.integers(self.cap))
        self.img[slot] = self.base[b]
        self.img[slot, :, y0:y1, x0:x1] = torch.from_numpy(np.ascontiguousarray(w)).to(self.dev).permute(2, 0, 1)
        self.msk[slot] = torch.from_numpy(m).to(self.dev)
        self.bi[slot], self.kind[slot] = b, kind
        self.rect[slot] = torch.tensor(rect, device=self.dev)
        self.n = min(self.cap, self.n + 1)

    def blobs(self, B):
        """Taches lisses aleatoires (bruit basse resolution agrandi puis seuille)."""
        g = int(self.rng.choice([3, 4, 6, 8, 12]))
        n = F.interpolate(torch.randn(B, 1, g, g, device=self.dev), size=(S, S), mode="bicubic", align_corners=False)[:, 0]
        n = (n - n.mean((1, 2), keepdim=True)) / (n.std((1, 2), keepdim=True) + 1e-6)
        frac = torch.empty(B, device=self.dev).uniform_(0.08, 0.6)
        return n > torch.special.ndtri(1 - frac)[:, None, None]

    def batch(self, B, debug=False):
        dev = self.dev
        idx = torch.randint(0, self.n, (B,), device=dev)
        r = self.rect[idx]
        rect = (self.yy >= r[:, 0, None, None]) & (self.yy < r[:, 1, None, None]) & \
               (self.xx >= r[:, 2, None, None]) & (self.xx < r[:, 3, None, None])
        blob = self.blobs(B)
        u = torch.rand(B, device=dev)
        m = self.msk[idx]                                                              # 72 % : masque de l'element
        m = torch.where(((u >= 0.72) & (u < 0.80))[:, None, None], rect, m)            # rectangle entier
        m = torch.where(((u >= 0.80) & (u < 0.92))[:, None, None], rect & ~blob, m)    # rectangle troue
        m = torch.where((u >= 0.92)[:, None, None], rect & blob, m)                    # tache (objet transplante)
        own = self.base[self.bi[idx]]
        alt = self.base[torch.randint(0, len(self.base), (B,), device=dev)]
        foreign = torch.rand(B, device=dev) < self.pf[self.kind[idx]]
        outside = torch.where(foreign[:, None, None, None], alt, own)                  # contenu hors masque
        x = torch.where(m[:, None], self.img[idx], outside).float().div_(255)
        y = (F.avg_pool2d(m[:, None].float(), 4)[:, 0] >= 0.5).float()                 # regle officielle 512 -> 128
        if debug:
            return x, y, m, idx, foreign
        return x, y


# ----------------------------------------------------------------------------- modele
class EncResNet(nn.Module):
    """ResNet-50 du decodeur TrustMark, applique en pleine resolution (4 echelles : pas 4, 8, 16, 32)."""
    chs = [256, 512, 1024, 2048]

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
        out = []
        for k in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(n, k)(x)
            out.append(x)
        return out


class EncConvNeXt(nn.Module):
    """ConvNeXt-tiny de l'extracteur PixelSeal, applique en pleine resolution (4 echelles)."""
    chs = [96, 192, 384, 768]

    def __init__(self, cnx):
        super().__init__()
        self.cnx = cnx

    def forward(self, x):
        x = x * 2 - 1
        out = []
        for i in range(4):
            x = self.cnx.stages[i](self.cnx.downsample_layers[i](x))
            out.append(x)
        return out


class Seg(nn.Module):
    """Encodeur partage applique a DEUX echelles (image 512 et image reduite a 256 = la vue native des extracteurs)
    + decodeur FPN leger -> logits 128 x 128 (une case = 4 x 4 pixels)."""

    def __init__(self, enc, d=128):
        super().__init__()
        self.enc = enc
        self.lat1 = nn.ModuleList([nn.Conv2d(c, d, 1) for c in enc.chs])     # vue 512 : pas 4, 8, 16, 32
        self.lat2 = nn.ModuleList([nn.Conv2d(c, d, 1) for c in enc.chs])     # vue 256 : pas 8, 16, 32, 64
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.ReLU(inplace=True))
                                     for _ in range(5)])
        self.head = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.ReLU(inplace=True), nn.Conv2d(d, 1, 1))

    @staticmethod
    def _to(a, size):
        a = a.float()
        return a if a.shape[-2:] == size else F.interpolate(a, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        f1 = self.enc(x)
        f2 = self.enc(F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True))
        lv = [self.lat1[0](f1[0]).float()]
        for i in (1, 2, 3):
            a = self.lat1[i](f1[i]).float()
            lv.append(a + self._to(self.lat2[i - 1](f2[i - 1]), a.shape[-2:]))
        lv.append(self.lat2[3](f2[3]).float())
        p = None
        for i in range(4, -1, -1):
            l = lv[i] if p is None else lv[i] + self._to(p, lv[i].shape[-2:])
            p = self.smooth[i](l)
        out = self.head(p).float()
        return self._to(out, (x.shape[-2] // 4, x.shape[-1] // 4))[:, 0]


def build_model(method, wms):
    if method == "pixelseal":
        enc = EncConvNeXt(copy.deepcopy(wms["pixelseal"].det.convnext))
    else:   # trustmark : son propre ResNet ; stegastamp (ONNX, pas de poids torch) : on part du ResNet TrustMark
        enc = EncResNet(copy.deepcopy(wms["trustmark"].net))
    for p in enc.parameters():
        p.requires_grad_(True)
    model = Seg(enc.float()).to(DEV).to(memory_format=torch.channels_last)
    set_train(model)                    # BatchNorm fige des maintenant (ne pas polluer ses statistiques)
    if AMP["on"]:                       # essai : un aller-retour en bf16 ; sinon on reste en float32
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(torch.rand(2, 3, 256, 256, device=DEV))
            o.float().mean().backward()
            assert o.shape == (2, 64, 64), o.shape
            model.zero_grad(set_to_none=True)
        except Exception as e:           # noqa: BLE001
            log("precision mixte bf16 indisponible, repli float32 :", repr(e)[:200])
            AMP["on"] = False
            model.zero_grad(set_to_none=True)
    return model


def set_train(model):
    model.train()
    for m in model.modules():           # BatchNorm fige (statistiques de l'extracteur d'origine)
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def seg_loss(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)

    def dice(a, b):
        return 1 - (2 * (a * b).sum((1, 2)) + 1) / (a.sum((1, 2)) + b.sum((1, 2)) + 1)
    return bce + 0.5 * (dice(p, y) + dice(1 - p, 1 - y)).mean()


def make_opt(model, lr_enc, lr_dec):
    enc = list(model.enc.parameters())
    dec = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    return torch.optim.AdamW([{"params": enc, "lr": lr_enc}, {"params": dec, "lr": lr_dec}], weight_decay=1e-4), [lr_enc, lr_dec]


def train_loop(model, opt, base_lrs, secs, batch_fn, tag, after_step=None, log_every=100):
    set_train(model)
    t0, step, run = time.time(), 0, None
    while True:
        el = time.time() - t0
        if el >= secs:
            break
        s = min(1.0, (step + 1) / 50) * max(0.02, 0.5 * (1 + math.cos(math.pi * el / secs)))   # echauffement + cosinus
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * s
        x, y = batch_fn()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP["on"]):
            logit = model(x.contiguous(memory_format=torch.channels_last))
        loss = seg_loss(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        step += 1
        if after_step is not None:
            after_step(step)
        if step % log_every == 0:
            log(f"  [{tag}] pas {step} ({el:.0f}s) perte lissee {run:.4f}")
    log(f"  [{tag}] fin : {step} pas en {time.time() - t0:.0f}s, perte lissee {run if run is not None else float('nan'):.4f}")
    return step


@torch.no_grad()
def predict(model, imgs_np, bs=16):
    model.eval()
    out = []
    for i in range(0, len(imgs_np), bs):
        x = torch.from_numpy(imgs_np[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float().div(255)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP["on"]):
            out.append(torch.sigmoid(model(x.contiguous(memory_format=torch.channels_last)).float()).cpu())
    return torch.cat(out).numpy()


def miou(probs, gts, t=0.5):
    return float(np.mean([miou_image(p >= t, g) for p, g in zip(probs, gts)]))


def report(tag, probs, gts, full=None):
    ts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sc = [miou(probs, gts, t) for t in ts]
    log(f"{tag} mIoU officiel selon seuil : " + "  ".join(f"{t:.1f}:{s:.4f}" for t, s in zip(ts, sc)))
    if full is not None and full.any() and (~full).any():
        log(f"     detail au seuil 0.5 : image entiere watermarkee (n={int(full.sum())}) {miou(probs[full], gts[full]):.4f} | "
            f"patch (n={int((~full).sum())}) {miou(probs[~full], gts[~full]):.4f}")
    return dict(zip(map(str, ts), sc))


@torch.no_grad()
def synth_report(model, pool, tag, n_batches, bs):
    """Diagnostic : mIoU officiel sur des images SYNTHETIQUES (meme generateur que l'entrainement), par type d'element.
    Sert a separer "le modele n'a pas fini d'apprendre" de "le synthetique ne ressemble pas au reel"."""
    model.eval()
    sc, kd = [], []
    for _ in range(n_batches):
        x, y, _m, idx, _f = pool.batch(bs, debug=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP["on"]):
            p = torch.sigmoid(model(x.contiguous(memory_format=torch.channels_last)).float())
        p, g = (p >= 0.5).cpu().numpy(), y.cpu().numpy().astype(np.uint8)
        sc += [miou_image(u, v) for u, v in zip(p, g)]
        kd += pool.kind[idx].cpu().tolist()
    sc, kd = np.array(sc), np.array(kd)
    noms = ["image entiere", "patch rectangle", "forme d'objet"]
    log(f"{tag} mIoU sur SYNTHETIQUE (n={len(sc)}, seuil 0.5) : {sc.mean():.4f} | "
        + " | ".join(f"{noms[k]} {sc[kd == k].mean():.4f}" for k in range(3) if (kd == k).any()))


# ----------------------------------------------------------------------------- pipeline
def pipeline(a, data, wms, dry):
    tr, te, tr_img, te_img, tr_msk, tr_m128 = data
    method = a.method
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    cfg = dict(pool_init=a.pool_init, synth=a.synth_secs, ft=a.ft_secs, folds=5, cap=a.cap)
    if dry:
        cfg = dict(pool_init=6, synth=8, ft=4, folds=2, cap=64)
    i_tr = np.array([i for i, m in enumerate(tr) if m["method"] == method])
    i_te = np.array([i for i, m in enumerate(te) if m["method"] == method])
    o_tr = np.array([i for i, m in enumerate(tr) if m["method"] != method])
    o_te = np.array([i for i, m in enumerate(te) if m["method"] != method])
    if dry:
        i_tr, i_te, o_tr, o_te = i_tr[:20], i_te[:8], o_tr[:40], o_te[:40]
    # fonds = images des AUTRES methodes (aucun watermark de la methode etudiee) ; masques = vrais masques des autres methodes
    base_np = np.concatenate([tr_img[o_tr], te_img[o_te]])
    masks_np = tr_msk[o_tr]
    fulls = [bool(tr[i].get("full_watermark", False)) for i in o_tr]
    max_ar = 2.0 if method == "trustmark" else 16.0
    pool = Pool(wms[method], base_np, masks_np, fulls, max_ar, cfg["cap"], rng)
    t0 = time.time()
    while time.time() - t0 < cfg["pool_init"] or pool.n < 16:
        pool.add()
    log(f"reserve initiale : {pool.n} images en {time.time() - t0:.0f}s (embed moyen {1000 * np.mean(pool.t_embed):.0f} ms)")

    model = build_model(method, wms)
    if a.init and Path(a.init).exists():
        # reprise tolerante : on charge tout ce qui a le meme nom et la meme forme (l'ancien decodeur "lat" -> "lat1")
        ck = torch.load(a.init, map_location="cpu", weights_only=True)
        ck = {("lat1." + k[4:] if k.startswith("lat.") else k): v for k, v in ck.items()}
        cur = model.state_dict()
        ok = {k: v.to(cur[k].dtype) for k, v in ck.items() if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(ok, strict=False)
        log(f"poids initiaux charges : {a.init} ({len(ok)}/{len(cur)} tenseurs repris)")
    elif a.init:
        log("poids initiaux introuvables, on part de l'extracteur :", a.init)
    real_x, real_y, test_x = tr_img[i_tr], tr_m128[i_tr], te_img[i_te]
    names_tr = [tr[i]["image_name"] for i in i_tr]
    names_te = [te[i]["image_name"] for i in i_te]
    real_full = np.array([bool(tr[i].get("full_watermark", False)) for i in i_tr])

    # ---------------- Phase A : synthetique seulement
    lr_enc = 3e-4 if method == "stegastamp" else 1e-4
    opt, lrs = make_opt(model, lr_enc, 1e-3)
    state = {"next": time.time() + a.eval_every, "t0": time.time(), "emb": 0.0}

    def after(step):
        for _ in range(2):               # on renouvelle la reserve sans y passer plus de ~30 % du temps
            if state["emb"] < 0.3 * (time.time() - state["t0"]):
                t = time.time()
                pool.add()
                state["emb"] += time.time() - t
        if not dry and time.time() > state["next"]:
            p = predict(model, real_x)
            log(f"  [A] pas {step} reserve {pool.n} : mIoU VRAI train (seuil 0.5) = {miou(p, real_y):.4f}")
            set_train(model)
            state["next"] = time.time() + a.eval_every
    if cfg["synth"] > 0:
        train_loop(model, opt, lrs, cfg["synth"], lambda: pool.batch(a.bs), "A synth", after)
    p_tr_synth = predict(model, real_x)
    p_te_synth = predict(model, test_x)
    sc_synth = report(f"[{method}] PHASE A (synthetique seul), vrai train n={len(i_tr)} :", p_tr_synth, real_y, real_full)
    synth_report(model, pool, f"[{method}] PHASE A", 2 if dry else 25, a.bs)
    synth_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if not dry:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in synth_state.items()}, OUT / f"poids_{method}_{a.tag}.pt")
        np.savez_compressed(OUT / f"probs_{method}_{a.tag}.npz", train_names=np.array(names_tr), test_names=np.array(names_te),
                            train_synth=p_tr_synth.astype(np.float16), test_synth=p_te_synth.astype(np.float16))

    # ---------------- Phase B : affinage sur les vraies images, validation croisee (hors-pli = honnete)
    order = np.argsort(names_tr)
    fold = np.zeros(len(i_tr), dtype=np.int64)
    fold[order] = np.arange(len(i_tr)) % cfg["folds"]
    rx = torch.from_numpy(real_x).to(DEV).permute(0, 3, 1, 2).contiguous()
    ry = torch.from_numpy(real_y).to(DEV).float()
    oof = np.zeros((len(i_tr), 128, 128), np.float32)
    te_cv = np.zeros((len(i_te), 128, 128), np.float32)
    nb_real = a.bs // 2
    for k in range(cfg["folds"]):
        model.load_state_dict(synth_state)
        opt, lrs = make_opt(model, 0.5 * lr_enc, 3e-4)
        tri = torch.from_numpy(np.where(fold != k)[0]).to(DEV)
        vai = np.where(fold == k)[0]

        def mixed():
            j = tri[torch.randint(0, len(tri), (nb_real,), device=DEV)]
            xs, ys = pool.batch(a.bs - nb_real)
            return torch.cat([rx[j].float().div(255), xs]), torch.cat([ry[j], ys])
        train_loop(model, opt, lrs, cfg["ft"], mixed, f"B pli {k}", None, log_every=200)
        oof[vai] = predict(model, real_x[vai])
        te_cv += predict(model, test_x) / cfg["folds"]
        log(f"  [B] pli {k} : mIoU hors-pli (seuil 0.5) = {miou(oof[vai], real_y[vai]):.4f}  (synthetique seul sur ce pli : {miou(p_tr_synth[vai], real_y[vai]):.4f})")
    sc_cv = report(f"[{method}] PHASE B (CV {cfg['folds']} plis, hors-pli), vrai train n={len(i_tr)} :", oof, real_y, real_full)
    if not dry:
        np.savez_compressed(OUT / f"probs_{method}_{a.tag}.npz", train_names=np.array(names_tr), test_names=np.array(names_te),
                            train_synth=p_tr_synth.astype(np.float16), test_synth=p_te_synth.astype(np.float16),
                            train_oof=oof.astype(np.float16), test_cv=te_cv.astype(np.float16), folds=fold)
        (OUT / "logs" / f"scores_{method}_{a.tag}.json").write_text(json.dumps({"synth": sc_synth, "cv": sc_cv, "args": vars(a)}, indent=1))
    del pool, model, rx, ry
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--synth_secs", type=float, default=270)
    ap.add_argument("--ft_secs", type=float, default=38)
    ap.add_argument("--pool_init", type=float, default=35)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--eval_every", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--no_dry", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    log("arguments :", vars(a))
    t0 = time.time()
    data = load_data()
    log(f"donnees chargees en {time.time() - t0:.0f}s : train {data[2].shape}, test {data[3].shape}")
    wms = {a.method: load_watermark(a.method, device=DEV)}
    if a.method == "stegastamp":
        wms["trustmark"] = load_watermark("trustmark", device=DEV)     # sert seulement d'initialisation de l'encodeur
    log(f"modeles de watermark charges ({time.time() - t0:.0f}s)")
    if not a.no_dry:
        log("=== essai a blanc (budgets minuscules) : verifie tout le code avant le vrai calcul ===")
        pipeline(a, data, wms, dry=True)
    log("=== calcul reel ===")
    pipeline(a, data, wms, dry=False)
    log(f"termine en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
