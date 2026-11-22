#!/usr/bin/env python
"""v45 - Indice D ("soutien par les voisins", idee de Twin Machine Unlearning, arXiv 2408.11433, et du C-score) :
le gold ne reconnait bien une image retiree que si une image TRES ressemblante de la meme classe est restee dans ses
donnees. On rejoue 300 fois "il ne reste que K images par classe lourde" et on regarde, pour chaque image mise de cote,
si son plus proche voisin garde est de sa classe (plus proche voisin, pas centre de classe). -> v45/knn_support.pt"""
import torch
torch.set_num_threads(8); torch.manual_seed(0)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"; y = torch.load(V + "cache.pt")["y"]; n = len(y); f = torch.load(V + "cache_feats.pt")
cnt = torch.bincount(y, minlength=100); classes = [c for c in range(100) if cnt[c] > 0]; heavy_c = [c for c in classes if cnt[c] >= 125]; heavy = torch.isin(y, torch.tensor(heavy_c))
idx = {c: (y == c).nonzero().flatten() for c in classes}; out = {}
for key in ("v1.0", "v1.5"):
    X = f[key].float(); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True)
    for K in (10, 20):
        err = torch.zeros(n); mar = torch.zeros(n); ev = torch.zeros(n); R = 300
        for r in range(R):
            ref = torch.zeros(n, dtype=torch.bool)
            for c in classes:
                ii = idx[c]; k = K if c in heavy_c else max(8, len(ii) // 2); ref[ii[torch.randperm(len(ii))[:k]]] = True
            te = (heavy & ~ref).nonzero().flatten(); rf = ref.nonzero().flatten(); S = X[te] @ X[rf].T; same = y[te][:, None] == y[rf][None, :]
            own = torch.where(same, S, torch.full_like(S, -9)).topk(3, 1).values.mean(1); oth = torch.where(~same, S, torch.full_like(S, -9)).topk(3, 1).values.mean(1)
            err[te] += (oth > own).float(); mar[te] += own - oth; ev[te] += 1
        out[f"{key}_K{K}"] = {"err": err / ev.clamp_min(1), "marge": mar / ev.clamp_min(1)}
        print(f"traits {key}, K={K} : taux d'erreur moyen (classes lourdes) {(err / ev.clamp_min(1))[heavy].mean():.3f} | par classe : " + " ".join(f"{c}:{(err / ev.clamp_min(1))[y == c].mean():.2f}" for c in heavy_c), flush=True)
    # densite relative statique : similarite au 5e voisin de meme classe - similarite au plus proche voisin d'une autre classe
    S = X @ X.T; S.fill_diagonal_(-9); same = y[:, None] == y[None, :]
    out[f"{key}_dens"] = torch.where(same, S, torch.full_like(S, -9)).topk(5, 1).values[:, 4] - torch.where(~same, S, torch.full_like(S, -9)).max(1).values
torch.save(out, V + "knn_support.pt")
P = torch.load(V + "indices.pt"); H = heavy
for k_ in ("v1.5_K20", "v1.0_K20", "v1.5_K10"):
    m = out[k_]["marge"]; print(f"correlation (classes lourdes) de l'indice D [{k_}] avec : " + " ".join(f"{a}:{torch.corrcoef(torch.stack([m[H], P[a][H].float()]))[0, 1]:.2f}" for a in ("vues16", "vues192", "prof6", "peu-ex", "tot2", "bruit0.06", "adv eps")))
for k_ in ("v1.5_dens", "v1.0_dens"):
    m = out[k_]; print(f"correlation de la densite relative [{k_}] avec : " + " ".join(f"{a}:{torch.corrcoef(torch.stack([m[H], P[a][H].float()]))[0, 1]:.2f}" for a in ("vues16", "vues192", "prof6", "peu-ex", "tot2")))
