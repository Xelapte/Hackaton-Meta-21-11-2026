"""Entraîne le modèle original (sur tout CIFAR-10) ou le réentraînement de référence
(sans le forget set).

Usage, depuis la racine du dépôt :
    python scrub/train_original.py --data <dossier_projet>/data --out results/scrub/original.pt
    python scrub/train_original.py --data ... --retrain --mode class \
        --out results/scrub/retrain_class.pt
    python scrub/train_original.py --fake --epochs 1 --out /tmp/test.pt   # test sur CPU
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scrub.cifar import (error, load_cifar, loader, resnet18_cifar,  # noqa: E402
                         split_forget, train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--augment", action="store_true",
                    help="augmentation de données (désactivée dans le papier)")
    ap.add_argument("--retrain", action="store_true", help="entraîner sans le forget set")
    ap.add_argument("--mode", default="class", choices=["class", "selective"])
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--n_forget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 0 if args.fake else 8
    tr, te = load_cifar(args.data, augment=args.augment, fake=args.fake)
    idx = None
    if args.retrain:
        _, idx = split_forget(tr, args.mode, args.target, args.n_forget, args.seed)
    model = train(resnet18_cifar(), loader(tr, idx, 128, True, workers), args.epochs,
                  args.lr, dev)
    print(f"Erreur de test : {error(model, loader(te, workers=workers), dev):.4f}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("Sauvegardé :", args.out)


if __name__ == "__main__":
    main()
