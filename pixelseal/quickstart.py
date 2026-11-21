"""Pixel Seal : tatouer des images, les attaquer, relire le message (baseline).

Usage, depuis la racine du dépôt d'équipe (le dossier ckpts/ des poids doit être
dans le dossier courant, voir la fiche bootstrap Pixel Seal) :
    python pixelseal/quickstart.py --images <dossier_images> [--scaling_w 0.2]
Écrit une ligne dans le journal des scores (SCORES_CSV) (score = bits justes moyens sur les attaques).
"""
import argparse
import io
import pathlib
import sys
from math import comb, log10

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import videoseal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.scores import log_score  # noqa: E402


def p_value(n_bits, n_err):
    """Proba qu'une image NON tatouée donne au plus n_err bits faux (fiche 1, 3.4)."""
    return sum(comb(n_bits, k) for k in range(n_err + 1)) / 2 ** n_bits


def jpeg(img, quality):
    buf = io.BytesIO()
    TF.to_pil_image(img[0]).save(buf, format="JPEG", quality=quality)
    return TF.to_tensor(Image.open(buf)).unsqueeze(0)


def crop_area(img, area):
    h, w = img.shape[-2:]
    s = area ** 0.5
    return TF.center_crop(img, [max(1, int(h * s)), max(1, int(w * s))])


ATTACKS = {  # images 1x3xHxW dans [0, 1], sur CPU
    "identite": lambda im: im,
    "jpeg_50": lambda im: jpeg(im, 50),
    "crop_50pct": lambda im: crop_area(im, 0.5),
    "luminosite_1.5": lambda im: TF.adjust_brightness(im, 1.5).clamp(0, 1),
    "flou_9": lambda im: TF.gaussian_blur(im, 9),
    "combinee": lambda im: jpeg(TF.adjust_brightness(crop_area(im, 0.5), 0.5), 40),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dossier d'images de test")
    ap.add_argument("--model", default="pixelseal", help="carte : pixelseal, videoseal…")
    ap.add_argument("--scaling_w", type=float, default=None,
                    help="force du tatouage (défaut de la carte : 0.2)")
    ap.add_argument("--max_images", type=int, default=50)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = videoseal.load(args.model).to(dev).eval()
    if args.scaling_w is not None:
        model.blender.scaling_w = args.scaling_w
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in pathlib.Path(args.images).iterdir()
                   if p.suffix.lower() in exts)[: args.max_images]
    if not paths:
        sys.exit(f"Aucune image trouvée dans {args.images}")

    accs = {name: [] for name in ATTACKS}
    logp = {name: [] for name in ATTACKS}
    psnrs = []
    with torch.no_grad():
        for p in paths:
            img = TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            msg = model.get_random_msg(1)  # 1 x nbits
            img_w = model.embed(img, msgs=msg)["imgs_w"].clamp(0, 1)
            mse = torch.mean((img_w - img) ** 2).item()
            psnrs.append(10 * log10(1 / max(mse, 1e-12)))
            target = msg[0].float().cpu()
            n_bits = target.numel()
            for name, attack in ATTACKS.items():
                att = attack(img_w.cpu()).to(dev)
                bits = (model.detect(att)["preds"][0, 1:] > 0).float().cpu()
                acc = (bits == target).float().mean().item()
                accs[name].append(acc)
                n_err = round((1 - acc) * n_bits)
                logp[name].append(-log10(max(p_value(n_bits, n_err), 1e-300)))

    print(f"{len(paths)} images, PSNR moyen {sum(psnrs) / len(psnrs):.1f} dB")
    print(f"{'attaque':16s} {'bits justes':>12s} {'-log10 p':>10s}")
    for name in ATTACKS:
        a = sum(accs[name]) / len(accs[name])
        lp = sum(logp[name]) / len(logp[name])
        print(f"{name:16s} {a:12.3f} {lp:10.1f}")
    score = sum(sum(v) / len(v) for v in accs.values()) / len(accs)
    log_score("pixelseal", {"model": args.model, "scaling_w": args.scaling_w,
                            "n_images": len(paths)}, round(score, 4),
              notes=f"psnr={sum(psnrs) / len(psnrs):.1f}")
    print("Score (bits justes moyens) :", round(score, 4))


if __name__ == "__main__":
    main()
