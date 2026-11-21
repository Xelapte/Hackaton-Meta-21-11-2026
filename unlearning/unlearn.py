"""Tâche 3 (Unlearning) : baseline de désapprentissage sans retain set.

Idée : le modèle « gold » n'a jamais vu les images à oublier ; sur elles, il se comporte
comme le modèle original sur des images inconnues : mêmes classes le plus souvent, mais
moins confiant (perte plus haute, plus dispersée). On imite cela en distillant le modèle
vers une version « adoucie » de ses propres prédictions sur les images à oublier
(softmax à température T > 1), en restant près des poids d'origine (ancre L2), pour
ne pas abîmer le reste (retain_score : précision et KL par rapport à l'original).

Usage (GPU, dans la file, avec le venv officiel) :
  bash slurm/soumettre.sh 1 ul_T2 --venv unlearning python unlearning/unlearn.py \
      --T 2.0 --steps 300 --lr 1e-5 --params all --out $OFFICIEL/unlearning/output/T2.pt
Puis soumettre le .pt avec le submission.py officiel (sur le nœud de login).
Test sans GPU ni données : python unlearning/unlearn.py --fake --steps 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def charger(args):
    """Renvoie (modèle original, modèle à modifier, images uint8 [N,224,224,3], preprocess)."""
    if args.fake:
        torch.manual_seed(0)
        def build():
            return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 7, stride=7), torch.nn.ReLU(),
                                       torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                       torch.nn.Linear(8, 100))
        orig = build()
        model = build(); model.load_state_dict(orig.state_dict())
        imgs = torch.randint(0, 256, (64, 224, 224, 3), dtype=torch.uint8)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        def prep(x):
            return (x.to(dev).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5
        return orig.to(dev).eval(), model.to(dev), imgs, prep
    off = Path(args.officiel) / "unlearning"
    sys.path.insert(0, str(off))
    import task_template as tt                     # fourni par les organisateurs
    orig = tt.load_model(off / "model.pt")
    model = tt.load_model(off / "model.pt")
    imgs = tt.load_forget_images(off / "forget_images.npz")
    return orig.eval(), model, imgs, tt.preprocess


def logits_de(model, imgs, prep, bs=128):
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=torch.cuda.is_available()):
        for i in range(0, len(imgs), bs):
            out.append(model(prep(imgs[i:i + bs])).float().cpu())
    return torch.cat(out)


def stats(logits, labels):
    loss = F.cross_entropy(logits, labels, reduction="none")
    p = logits.softmax(-1)
    return {"loss_mean": loss.mean().item(), "loss_std": loss.std().item(),
            "conf_mean": p.max(-1).values.mean().item(),
            "accord_avec_orig": (logits.argmax(-1) == labels).float().mean().item()}


def params_a_entrainer(model, quoi):
    for p in model.parameters():
        p.requires_grad_(quoi == "all")
    if quoi == "all":
        return [p for p in model.parameters()]
    noms = {"head": ("head",), "norm": ("norm", "head"), "last": ("blocks.11", "norm", "head")}[quoi]
    choisis = []
    for n, p in model.named_parameters():
        if any(k in n for k in noms):
            p.requires_grad_(True)
            choisis.append(p)
    if not choisis:                                # modèle factice : tout entraîner
        for p in model.parameters():
            p.requires_grad_(True)
        choisis = list(model.parameters())
    return choisis


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--officiel", default=os.environ.get("OFFICIEL", ""))
    ap.add_argument("--T", type=float, default=2.0, help="température de la cible (> 1 = adoucir)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--anchor", type=float, default=1e-2, help="poids de l'ancre L2 vers l'original")
    ap.add_argument("--params", default="all", choices=["all", "last", "norm", "head"])
    ap.add_argument("--out", default="submission.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    orig, model, imgs, prep = charger(args)
    dev = next(model.parameters()).device
    lo = logits_de(orig, imgs, prep)
    pseudo = lo.argmax(-1)                          # pas d'étiquettes : celles de l'original
    cible = (lo / args.T).softmax(-1)               # prédictions adoucies
    print("Original sur les images à oublier :", json.dumps(stats(lo, pseudo)))

    params = params_a_entrainer(model, args.params)
    ref = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    model.train()
    n = len(imgs)
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.bs,))
        x = prep(imgs[idx])
        if torch.rand(1).item() < 0.5:
            x = x.flip(-1)                          # petite augmentation
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(x).float()
        kl = F.kl_div(out.log_softmax(-1), cible[idx].to(dev), reduction="batchmean")
        ancre = sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
        loss = kl + args.anchor * ancre
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"pas {step:4d}  KL {kl.item():.4f}  ancre {ancre.item():.3e}")

    ls = logits_de(model, imgs, prep)
    s = stats(ls, pseudo)
    print("Après désapprentissage :", json.dumps(s))
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values()), "NaN ou Inf dans les poids"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    meta = {"args": vars(args), "apres": s, "duree_s": round(time.time() - t0)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"Écrit : {args.out} ({os.path.getsize(args.out) / 1e6:.0f} Mo) et {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
