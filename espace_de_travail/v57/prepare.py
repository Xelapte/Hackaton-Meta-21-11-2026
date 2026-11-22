#!/usr/bin/env python
"""v57 - Rassemble les images fabriquees a partir de texte et fait le bilan : quelles classes sont couvertes, et est-ce que
les objets dont on connait la classe (renard = 0, lanternes = 5-8, ...) tombent bien dans la bonne classe ?
Sorties : gen/txt_all.npy (images gardees, melangees, au plus CAP par classe) + gen/txt_all.pt (reponses de l'original)."""
import sys, numpy as np, torch
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v57/"; CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 160; torch.manual_seed(0)
X, Z, OBJ = [], [], []
for i in (0, 1):
    m = torch.load(V + f"gen/txt_{i}_meta.pt"); x = np.load(V + f"gen/txt_{i}.npy"); k = m["keep"]; assert len(x) == int(k.sum())
    X.append(torch.from_numpy(x)); Z.append(m["z"][k].float()); OBJ.append(m["obj"][k]); vocab = m["vocab"]
    zz = m["z"].float(); print(f"partie {i} : {len(zz)} generees, {int(k.sum())} gardees ; confiance moyenne de l'original {zz.softmax(1).max(1).values.mean():.2f}")
X = torch.cat(X); Z = torch.cat(Z); OBJ = torch.cat(OBJ); pred = Z.argmax(1); conf = Z.softmax(1).max(1).values
cnt = torch.bincount(pred, minlength=100); print(f"total garde {len(X)} | classes avec >= 1 image : {(cnt > 0).sum().item()}/100, >= 20 : {(cnt >= 20).sum().item()}, >= 60 : {(cnt >= 60).sum().item()}")
print("images par classe :", " ".join(f"{c}:{cnt[c].item()}" for c in range(100)))
yf = torch.load("/p/scratch/training2666/huang25/Epi84/unlearning/v45/cache.pt")["y"]; cf = torch.bincount(yf, minlength=100)
print("classes du forget set SANS image fabriquee :", [c for c in range(100) if cf[c] > 0 and cnt[c] < 5]); print("classes absentes du forget set AVEC >= 20 images fabriquees :", sum(1 for c in range(100) if cf[c] == 0 and cnt[c] >= 20), "sur", int((cf == 0).sum()))
for c in range(100):
    m = pred == c
    if m.sum() == 0: continue
    ob = torch.bincount(OBJ[m], minlength=len(vocab)); top = ob.topk(3)
    print(f"classe {c:2d} ({'oubli ' + str(cf[c].item()) if cf[c] > 0 else 'absente'}) : {int(m.sum()):4d} images, confiance {conf[m].mean():.2f} | " + ", ".join(f"{vocab[j]} ({v})" for v, j in zip(top.values.tolist(), top.indices.tolist()) if v > 0))
keep = torch.zeros(len(X), dtype=torch.bool)
for c in range(100):
    ii = (pred == c).nonzero().flatten(); ii = ii[conf[ii].argsort(descending=True)][:CAP]; keep[ii] = True
ii = keep.nonzero().flatten(); ii = ii[torch.randperm(len(ii))]
np.save(V + "gen/txt_all.npy", X[ii].numpy()); torch.save({"z": Z[ii].half(), "obj": OBJ[ii], "vocab": vocab}, V + "gen/txt_all.pt")
print(f"sauve gen/txt_all.npy : {len(ii)} images (au plus {CAP} par classe), {torch.bincount(pred[ii], minlength=100).gt(0).sum().item()} classes")
