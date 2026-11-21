"""Lance SCRUB (et SCRUB+R) sur le modèle original, puis mesure l'oubli.

Usage, depuis la racine du dépôt :
    python scrub/run_scrub.py --data <dossier_projet>/data \
        --original results/scrub/original.pt --mode class
    python scrub/run_scrub.py ... --retrain_ckpt results/scrub/retrain_class.pt
    python scrub/run_scrub.py --fake --original /tmp/test.pt --n_epochs 2   # test CPU
Mesures : erreurs test / retain / forget, et MIA de base (forget contre test de la
même classe ; idéal = 50 %). Écrit une ligne dans le journal des scores (SCORES_CSV).
"""
import argparse
import copy
import pathlib
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.scores import log_score  # noqa: E402
from scrub.cifar import (error, labels_of, load_cifar, loader,  # noqa: E402
                         predict, resnet18_cifar, split_forget)
from scrub.scrub import scrub  # noqa: E402


def mia(l_in, l_out, seed=0):
    """Régression logistique sur les losses, jeux équilibrés, 5 plis (fiche 3)."""
    n = min(len(l_in), len(l_out))
    X = np.clip(np.r_[l_in[:n], l_out[:n]], -400, 400).reshape(-1, 1)
    y = np.r_[np.ones(n), np.zeros(n)]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(LogisticRegression(), X, y, cv=cv).mean())


def evaluate(model, dls, dev):
    l_f, _, _ = predict(model, dls["forget"], dev)
    l_o, _, _ = predict(model, dls["test_target"], dev)
    return {"test": error(model, dls["test"], dev),
            "retain": error(model, dls["retain_eval"], dev),
            "forget": error(model, dls["forget"], dev),
            "mia": mia(l_f, l_o)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--original", required=True)
    ap.add_argument("--retrain_ckpt", default=None,
                    help="checkpoint du réentraînement, pour comparer (la cible en UP)")
    ap.add_argument("--mode", default="class", choices=["class", "selective"])
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--n_forget", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=2)
    ap.add_argument("--n_epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs_forget", type=int, default=512)
    ap.add_argument("--bs_retain", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = 0 if args.fake else 8
    tr, te = load_cifar(args.data, fake=args.fake)
    forget, retain = split_forget(tr, args.mode, args.target, args.n_forget, args.seed)
    y_te = labels_of(te)
    te_t = np.flatnonzero(y_te == args.target)
    # validation de même distribution que le forget set : la moitié du test de la classe
    rng = np.random.RandomState(args.seed)
    rng.shuffle(te_t)
    val_t, test_t = te_t[: len(te_t) // 2], te_t[len(te_t) // 2:]
    eval_retain = rng.choice(retain, min(5000, len(retain)), replace=False)
    dls = {"forget": loader(tr, forget, 256, workers=w),
           "retain_eval": loader(tr, eval_retain, 256, workers=w),
           "test": loader(te, workers=w),
           "test_target": loader(te, test_t, 256, workers=w),
           "val_target": loader(te, val_t, 256, workers=w)}

    original = resnet18_cifar()
    original.load_state_dict(torch.load(args.original, map_location="cpu"))
    original.to(dev)
    res = {"original": evaluate(original, dls, dev)}
    if args.retrain_ckpt:
        ref_model = resnet18_cifar()
        ref_model.load_state_dict(torch.load(args.retrain_ckpt, map_location="cpu"))
        res["retrain"] = evaluate(ref_model.to(dev), dls, dev)

    student, ckpts = scrub(original,
                           loader(tr, forget, args.bs_forget, True, w),
                           loader(tr, retain, args.bs_retain, True, w),
                           max_steps=args.max_steps, n_epochs=args.n_epochs,
                           alpha=args.alpha, gamma=args.gamma, lr=args.lr)
    res["scrub"] = evaluate(student, dls, dev)

    # SCRUB+R : checkpoint dont l'erreur d'oubli est la plus proche de la référence
    ref = error(student, dls["val_target"], dev)
    probe = copy.deepcopy(student)
    gaps = []
    for ck in ckpts:
        probe.load_state_dict(ck)
        gaps.append(abs(error(probe, dls["forget"], dev) - ref))
    probe.load_state_dict(ckpts[int(np.argmin(gaps))])
    res["scrub_r"] = evaluate(probe, dls, dev)

    print(f"{'':10s} {'test':>7s} {'retain':>7s} {'forget':>7s} {'MIA':>7s}")
    for k, r in res.items():
        print(f"{k:10s} {r['test']:7.3f} {r['retain']:7.3f} {r['forget']:7.3f} "
              f"{r['mia']:7.3f}")
    cfg = {k: getattr(args, k) for k in ["mode", "max_steps", "n_epochs", "alpha",
                                         "gamma", "lr", "seed"]}
    for k in ["scrub", "scrub_r"]:
        log_score("scrub", dict(cfg, variante=k), round(res[k]["forget"], 4),
                  notes=f"test={res[k]['test']:.4f} retain={res[k]['retain']:.4f} "
                        f"mia={res[k]['mia']:.4f}")


if __name__ == "__main__":
    main()
