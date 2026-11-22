#!/usr/bin/env python
"""v45 - Simulation "peu d'exemples" : le gold n'a garde que quelques images de chaque classe lourde. On rejoue cette situation
dans l'espace des traits de l'original (moyenne sur 16 vues modifiees = sans par-coeur) : a chaque tirage on garde K images
au hasard par classe lourde (et la moitie des images des classes partielles) comme "exemples connus", on calcule le centre
de chaque classe, et on classe les autres images par centre le plus proche. Une image souvent mal classee = image que le
gold a de bonnes chances de rater. Sortie : taux d'erreur et marge moyenne par image -> v45/peu_exemples.pt"""
import sys, torch
torch.set_num_threads(8); torch.manual_seed(0)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v45/"; y = torch.load(V + "cache.pt")["y"]; n = len(y); f = torch.load(V + "cache_feats.pt")
cnt = torch.bincount(y, minlength=100); classes = [c for c in range(100) if cnt[c] > 0]; heavy_c = [c for c in classes if cnt[c] >= 125]; heavy = torch.isin(y, torch.tensor(heavy_c))
idx = {c: (y == c).nonzero().flatten() for c in classes}; cl_t = torch.tensor(classes); pos = {c: i for i, c in enumerate(classes)}; ypos = torch.tensor([pos[int(c)] for c in y])
out = {}
for key in ("v1.0", "v1.5"):
    X = f[key].float(); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True)
    for K in (10, 20, 40):
        err = torch.zeros(n); mar = torch.zeros(n); ev = torch.zeros(n); R = 300
        for r in range(R):
            ref = torch.zeros(n, dtype=torch.bool); C = []
            for c in classes:
                ii = idx[c]; k = K if c in heavy_c else max(8, len(ii) // 2); sel = ii[torch.randperm(len(ii))[:k]]; ref[sel] = True; m = X[sel].mean(0); C.append(m / m.norm())
            C = torch.stack(C); te = heavy & ~ref; S = X[te] @ C.T; yy = ypos[te]; ar = torch.arange(len(yy))
            own = S[ar, yy]; S2 = S.clone(); S2[ar, yy] = -9; oth = S2.max(1).values
            err[te] += (oth > own).float(); mar[te] += own - oth; ev[te] += 1
        pe = err / ev.clamp_min(1); mm = mar / ev.clamp_min(1); out[f"{key}_K{K}"] = {"err": pe, "marge": mm}
        print(f"traits {key}, {K} exemples par classe lourde : taux d'erreur moyen sur les classes lourdes {pe[heavy].mean():.3f} | par classe : " + " ".join(f"{c}:{pe[y == c].mean():.2f}" for c in heavy_c), flush=True)
torch.save(out, V + "peu_exemples.pt")
