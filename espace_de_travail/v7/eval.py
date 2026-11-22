#!/usr/bin/env python
"""v6 - Evaluation / classement automatique de modeles, avec baisse du biais des classes entieres (beta).
Aucun entrainement : pour chaque modele et chaque beta, on mesure sur GPU (comme le serveur : eval + fp16) :
  - acc / loss (T=2) sur le forget set : tout, classes entieres, classes partielles, vues modifiees ;
  - derive vs l'original sur des mosaiques (KL T=2 + accord), sur 2 jeux :
      mos_all  = mosaiques de toutes les images du forget set (comme v1-v5),
      mos_part = mosaiques SANS les images des classes entieres (si le jeu de controle cache ne contient pas
                 ces classes, c'est la mesure la plus representative du retain).
Usage : python eval.py <modele1.pt> [modele2.pt ...] --betas 0 4 8 12
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unlearn as U  # noqa: E402  (chdir vers unlearning/, import task_template)

FULL = [0, 6, 7, 8, 29, 30, 31, 33, 38]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 4.0, 8.0, 12.0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "logs" / "eval.jsonl"))
    args = ap.parse_args()

    orig = U.load_model()
    forget = U.load_forget_images()
    N = len(forget)
    lo_clean = U.logits_u8(orig, forget)
    y = lo_clean.argmax(1)
    full_cls = torch.tensor(FULL, device=y.device)
    full = torch.isin(y, full_cls)
    g = torch.Generator().manual_seed(123)
    mos_all = torch.randint(0, N, (1024, 4), generator=g)
    part_idx = (~full).nonzero().flatten().cpu()
    mos_part = part_idx[torch.randint(0, len(part_idx), (1024, 4), generator=g)]
    lo_mos_all = U.logits_from_unit(orig, lambda i, j: U.mosaic(forget, mos_all[i:j]), 1024)
    lo_mos_part = U.logits_from_unit(orig, lambda i, j: U.mosaic(forget, mos_part[i:j]), 1024)
    print(f"{int(full.sum())} images classes entieres, {int((~full).sum())} partielles", flush=True)

    def st(lo, m=None):
        lo_, y_ = (lo, y) if m is None else (lo[m], y[m])
        ce = F.cross_entropy(lo_ / 2, y_, reduction="none")
        return round((lo_.argmax(1) == y_).float().mean().item(), 3), round(ce.mean().item(), 2), round(ce.std().item(), 2)

    def shift(lo, beta):
        lo = lo.clone()
        lo[:, full_cls] -= beta
        return lo

    hdr = (f"{'modele':28} {'beta':>4} | {'acc':>5} {'L2':>5} {'sd':>5} | {'acc_full':>8} {'acc_part':>8} | "
           f"{'acc_aug':>7} | {'kl_mos_all':>10} {'agr':>5} | {'kl_mos_part':>11} {'agr':>5}")
    print(hdr, flush=True)
    outf = open(args.out, "a")
    for path in args.models:
        model = U.load_model(Path(path).resolve())
        lc = U.logits_u8(model, forget)
        la = U.logits_u8(model, forget, aug=1.0, seed=1)
        lma = U.logits_from_unit(model, lambda i, j: U.mosaic(forget, mos_all[i:j]), 1024)
        lmp = U.logits_from_unit(model, lambda i, j: U.mosaic(forget, mos_part[i:j]), 1024)
        for b in args.betas:
            c, a, ma, mp = shift(lc, b), shift(la, b), shift(lma, b), shift(lmp, b)
            r = {
                "model": path, "beta": b,
                "all": st(c), "full": st(c, full), "part": st(c, ~full), "aug": st(a),
                "kl_mos_all": round(U.kl_t2(lo_mos_all, ma), 3),
                "agree_mos_all": round((ma.argmax(1) == lo_mos_all.argmax(1)).float().mean().item(), 3),
                "kl_mos_part": round(U.kl_t2(lo_mos_part, mp), 3),
                "agree_mos_part": round((mp.argmax(1) == lo_mos_part.argmax(1)).float().mean().item(), 3),
            }
            outf.write(json.dumps(r) + "\n")
            name = "/".join(Path(path).parts[-3:])[-28:]
            print(f"{name:28} {b:4.0f} | {r['all'][0]:5.3f} {r['all'][1]:5.2f} {r['all'][2]:5.2f} | "
                  f"{r['full'][0]:8.3f} {r['part'][0]:8.3f} | {r['aug'][0]:7.3f} | "
                  f"{r['kl_mos_all']:10.3f} {r['agree_mos_all']:5.3f} | {r['kl_mos_part']:11.3f} {r['agree_mos_part']:5.3f}",
                  flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
