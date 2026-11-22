"""v80 : tests du 1er ordre sur les sorties de score4.py.
Pour chaque endpoint et chaque prompt : cellules (token precedent a, token courant v) parmi les tokens frequents ;
exces D[a,v] = somme_t 1(x_{t-1}=a) (1(x_t=v) - p_t(v)), variance W[a,v] = somme_t 1(x_{t-1}=a) p_t(v)(1-p_t(v)) (martingale exacte
sous 'modele de base, T0'). chi2 = somme D^2/W sur les cellules de variance >= wmin. Aussi : exces marginal par token (a = tout)."""
import sys, math, torch, collections
wmin = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
res = torch.load(sys.argv[1]); suivis = res["suivis"]
eps = sorted({r["endpoint"] for r in res["rep"]}); ids = sorted({r["id"] for r in res["rep"]})
tot = {e: [0.0, 0, 0.0, 0] for e in eps}
for pid in ids:
    S = suivis[pid]; K = len(S); pos = {int(t): i for i, t in enumerate(S)}
    # tokens frequents (poole, sans regarder l'endpoint) : ceux qui font >= 2 % des tokens
    cnt = torch.zeros(K)
    for r in res["rep"]:
        if r["id"] == pid:
            cnt += torch.bincount(torch.tensor([pos[int(t)] for t in r["x"]]), minlength=K).float()
    F = (cnt / cnt.sum() >= 0.02).nonzero().flatten().tolist()
    line = []
    for e in eps:
        D1 = torch.zeros(len(F), dtype=torch.float64); W1 = torch.zeros(len(F), dtype=torch.float64)
        D2 = torch.zeros(len(F), len(F), dtype=torch.float64); W2 = torch.zeros(len(F), len(F), dtype=torch.float64); n = nr = 0
        for r in res["rep"]:
            if r["id"] != pid or r["endpoint"] != e:
                continue
            xi = torch.tensor([pos[int(t)] for t in r["x"]]); P = r["P"].double()[:, F]       # [n, |F|]
            ind = torch.stack([(xi == f).double() for f in F], 1)                           # [n, |F|]
            dv = ind - P; wv = P * (1 - P)
            D1 += dv.sum(0); W1 += wv.sum(0); n += len(xi); nr += 1
            prev = ind[:-1]                                                                  # token precedent = F[a]
            D2 += prev.T @ dv[1:]; W2 += prev.T @ wv[1:]
        if nr == 0:
            continue
        m1 = W1 >= wmin; c1 = float((D1[m1] ** 2 / W1[m1]).sum()); k1 = int(m1.sum())
        m2 = W2 >= wmin; c2 = float((D2[m2] ** 2 / W2[m2]).sum()); k2 = int(m2.sum())
        tot[e][0] += c1; tot[e][1] += k1; tot[e][2] += c2; tot[e][3] += k2
        line.append(f"{e[-2:]}: n={n:5d} uni {c1:5.1f}/{k1} cond {c2:6.1f}/{k2}")
    print(pid[:9].ljust(9), "|", "  ||  ".join(line))
print("TOTAL    ", "|", "  ||  ".join(f"{e[-2:]}: uni {t[0]:5.1f}/{t[1]} (z={(t[0] - t[1]) / math.sqrt(2 * max(1, t[1])):+.1f}) cond {t[2]:6.1f}/{t[3]} (z={(t[2] - t[3]) / math.sqrt(2 * max(1, t[3])):+.1f})" for e, t in tot.items()))
