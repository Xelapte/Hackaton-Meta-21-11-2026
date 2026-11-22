#!/usr/bin/env python
"""v8 - Enregistre une fois pour toutes les reponses (logits) de plusieurs modeles. Aucun entrainement.
Pour chaque modele : logits sur
  - les 3500 images exactes,
  - 2 vues legerement modifiees (aug 0.5) et 2 vues plus modifiees (aug 1.0), memes vues pour tous les modeles,
  - 1024 mosaiques de toutes les images + 1024 mosaiques sans les classes entieres (proxy du retain).
Sortie : cache/<nom>.pt. Ensuite le simulateur (sim.py) tourne sur CPU en quelques secondes.
Usage : python cache.py nom1=/chemin/absolu/modele1.pt nom2=/chemin/absolu/modele2.pt ...
"""
import sys
from pathlib import Path

import torch

VDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VDIR))
import unlearn as U  # noqa: E402  (fait chdir vers unlearning/ -> toujours passer des chemins ABSOLUS)

FULL = [0, 6, 7, 8, 29, 30, 31, 33, 38]


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    forget = U.load_forget_images()
    N = len(forget)
    orig = U.load_model()
    y = U.logits_u8(orig, forget).argmax(1)
    full = torch.isin(y, torch.tensor(FULL, device=y.device))
    g = torch.Generator().manual_seed(123)
    mos_all = torch.randint(0, N, (1024, 4), generator=g)
    part_idx = (~full).nonzero().flatten().cpu()
    mos_part = part_idx[torch.randint(0, len(part_idx), (1024, 4), generator=g)]
    torch.save({"y": y.cpu(), "full": full.cpu(), "full_cls": FULL}, VDIR / "cache" / "_meta.pt")

    def run(model):
        return {
            "clean": U.logits_u8(model, forget).cpu(),
            "aug05": torch.stack([U.logits_u8(model, forget, aug=0.5, seed=500 + k).cpu() for k in range(2)]),
            "aug10": torch.stack([U.logits_u8(model, forget, aug=1.0, seed=600 + k).cpu() for k in range(2)]),
            "mos_all": U.logits_from_unit(model, lambda i, j: U.mosaic(forget, mos_all[i:j]), 1024).cpu(),
            "mos_part": U.logits_from_unit(model, lambda i, j: U.mosaic(forget, mos_part[i:j]), 1024).cpu(),
        }

    torch.save(run(orig), VDIR / "cache" / "orig.pt")
    print("orig ok", flush=True)
    for name, path in specs:
        out = VDIR / "cache" / f"{name}.pt"
        if out.exists():
            print(f"{name} deja en cache", flush=True)
            continue
        model = U.load_model(Path(path))
        torch.save(run(model), out)
        print(f"{name} ok", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
