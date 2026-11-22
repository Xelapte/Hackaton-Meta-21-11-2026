"""Tâche 2 : décision finale (familles + modèles) à partir des sorties de score_local.py et des réponses d'identité.

1. Famille : modèle local candidat qui donne la log-proba moyenne par token la plus haute.
2. Dans chaque famille : LLR par token log p(base) - log p(fp8) sous le modèle de la famille, z = moyenne / erreur type.
   Sur les ref- : fp8 z = -2,71 (60 réponses) ; S plain/syst/tatouage/temp1 z = +2 à +4,7 ; persona +1,40 ; LoRA +1,2 à +3,3.
   Une persona (identité qui ne cite pas la famille) baisse le z sans être une variante : elle n'est jamais marquée.
   Règle : z < --seuil_z et identité normale -> variante, seule dans son modèle ; sinon -> modèle de base de la famille.
   (Les LoRA ne sont pas vues par ce test. Sur la structure des ref-, cette règle seule vaut 0,331.)
3. --calib (sorties de calibration.py, v2) : calib = (lp + H) / n sous le modèle de la famille.
   - déguisement « température 1,0 » : calib absolu < --seuil_temp (ep- : 33, 30, 31 ≈ 0 ; ref-08 -0,11) -> base ;
   - LoRA : écart apparié à la médiane de la famille, z < --seuil_calib, identité normale -> variante
     (ref- : LoRA 0,7 -2,84, LoRA 1,5 -5,09 ; persona -2,47, exclue par l'identité ; LoRA 0,3 non vue).

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
    ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--seuil_calib", type=float, default=-2.5)
    ap.add_argument("--seuil_temp", type=float, default=0.05)
    ap.add_argument("--exclure", default="A01_identite", help="prompts ignorés pour la calibration")
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
    cal = defaultdict(dict)                            # endpoint -> (id, repeat) -> calib (modèle de la famille, filtré plus bas)
    for f in args.calib:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["id"] not in args.exclure.split(",") and int(r["n"]) >= 5:
                cal[(r["endpoint"], r["modele"])][(r["id"], r["repeat"])] = (float(r["lp"]) + float(r["H"])) / int(r["n"])
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
        lignes.append([e, fam, z, moy[fam], persona])
    # calibration appariée, dans la famille
    zc, ca = {}, {}
    for e, fam, *_ in lignes:
        c = cal.get((e, fam))
        if not c:
            continue
        autres = [o for o, f2, *_ in lignes if f2 == fam and o != e and cal.get((o, fam))]
        ca[e] = sum(c.values()) / len(c)
        d = []
        for k, x in c.items():
            ref = sorted(v for o in autres for kk, v in cal[(o, fam)].items() if kk[0] == k[0])
            if len(ref) >= 3:
                d.append(x - ref[len(ref) // 2])
        m = sum(d) / len(d); et = (sum((y - m) ** 2 for y in d) / (len(d) - 1) / len(d)) ** 0.5
        zc[e] = m / et
    print(f"{'endpoint':8s} {'famille':14s} {'z_fp8':>6s} {'logp':>7s} {'calib':>6s} {'z_cal':>6s} persona temp -> modèle")
    rows = []
    for e, fam, z, l, p in sorted(lignes, key=lambda x: (x[1], -x[2])):
        temp = e in ca and ca[e] < args.seuil_temp
        lora = e in zc and zc[e] < args.seuil_calib
        v = not p and not temp and (z < args.seuil_z or lora)
        mod = f"{fam}_{e}" if v else f"{fam}_base"
        rows.append((e, fam, mod))
        print(f"{e:8s} {fam:14s} {z:+6.2f} {l:+7.3f} {ca.get(e, float('nan')):+6.3f} {zc.get(e, float('nan')):+6.2f} "
              f"{'oui' if p else '   '}     {'oui' if temp else '   '}  -> {mod}{'  (fp8 ?)' if v and z < args.seuil_z else ''}"
              f"{'  (LoRA ?)' if v and lora else ''}")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["endpoint", "family", "model"]); w.writerows(sorted(rows))
        print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
