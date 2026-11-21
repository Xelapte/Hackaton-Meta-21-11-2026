"""Tâche 2 : décision finale (familles + modèles) à partir des sorties de score_local.py et des réponses d'identité.

1. Famille : modèle local candidat qui donne la log-proba moyenne par token la plus haute.
2. Dans chaque famille : LLR par token log p(base) - log p(fp8) sous le modèle de la famille, z = moyenne / erreur type.
   Sur les ref- : fp8 z = -2,71 (60 réponses) ; S plain/syst/tatouage/temp1 z = +2 à +4,7 ; persona +1,40 ; LoRA +1,2 à +3,3.
   Une persona (identité qui ne cite pas la famille) baisse le z sans être une variante : elle n'est jamais marquée.
   Règle : z < --seuil_z et identité normale -> variante, seule dans son modèle ; sinon -> modèle de base de la famille.
   (Les LoRA ne sont pas vues par ce test : elles restent avec la base. Sur la structure des ref-, cette règle vaut
   ~0,33 de MCC « modèle », contre 0 pour « tout ensemble » ou « tout séparé ».)

  python met/decision.py --llr out/llr_ep_a.csv out/llr_ep_b.csv out/llr_ep_c.csv \\
      --reponses out/collecte_ep.jsonl --out out/soumission_me_v1.csv
"""
import argparse
import csv
import json
from collections import defaultdict

MOTS = {"Qwen": ("qwen", "alibaba"), "Llama": ("llama", "meta"), "SmolLM": ("smollm", "hugging")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llr", nargs="+", required=True)
    ap.add_argument("--reponses", required=True)
    ap.add_argument("--prefix", default="ep-")
    ap.add_argument("--seuil_z", type=float, default=0.5)
    ap.add_argument("--out")
    args = ap.parse_args()
    lp = defaultdict(lambda: defaultdict(dict))          # (endpoint, modèle) -> (id, repeat) -> variante -> (lp, n)
    for f in args.llr:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["endpoint"].startswith(args.prefix) and int(r["ntokens"]) > 0:
                lp[(r["endpoint"], r["modele"])][(r["id"], r["repeat"])][r["variante"]] = (float(r["logprob"]), int(r["ntokens"]))
    ident = defaultdict(list)
    for l in open(args.reponses, encoding="utf-8"):
        r = json.loads(l)
        if r["endpoint"].startswith(args.prefix) and r["id"].endswith("identite"):
            ident[r["endpoint"]].append((r["response"] or "").lower())
    eps = sorted({e for e, _ in lp}); modeles = sorted({m for _, m in lp})
    lignes = []
    for e in eps:
        moy = {m: sum(v["base"][0] for v in lp[(e, m)].values()) / sum(v["base"][1] for v in lp[(e, m)].values())
               for m in modeles}
        fam = max(moy, key=moy.get)
        llr = [(v["base"][0] - v["fp8"][0]) / v["base"][1] for v in lp[(e, fam)].values()]
        mu = sum(llr) / len(llr); et = (sum((x - mu) ** 2 for x in llr) / (len(llr) - 1) / len(llr)) ** 0.5
        z = mu / et
        cle = next(k for k in MOTS if fam.startswith(k))
        persona = 2 * sum(any(w in t for w in MOTS[cle]) for t in ident[e]) < len(ident[e])   # famille citée < 1 fois sur 2
        variante = z < args.seuil_z and not persona
        lignes.append((e, fam, z, moy[fam], persona, variante))
    print(f"{'endpoint':8s} {'famille':14s} {'z':>6s} {'logp':>7s} persona -> modèle")
    rows = []
    for e, fam, z, l, p, v in sorted(lignes, key=lambda x: (x[1], -x[2])):
        mod = f"{fam}_{e}" if v else f"{fam}_base"
        rows.append((e, fam, mod))
        print(f"{e:8s} {fam:14s} {z:+6.2f} {l:+7.3f} {'oui' if p else '   '}     -> {mod}")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["endpoint", "family", "model"]); w.writerows(sorted(rows))
        print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
