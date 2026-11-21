"""Tâche 2 : comparer chaque endpoint à un modèle de référence local (et à sa version fp8).

Idée (d'après RUT, arXiv 2506.06975) : avec le vrai modèle de base en local, on calcule la probabilité
que lui donne chaque réponse d'un endpoint. On fabrique aussi sa version fp8 (poids quantifiés en
float8_e4m3, échelle par tenseur, lm_head non quantifié, comme vLLM). Pour chaque réponse :
    LLR = log p_base(réponse) - log p_fp8(réponse)      (par token)
  - endpoint = modèle de base (même déguisé) -> LLR moyen > 0
  - endpoint = version fp8                    -> LLR moyen < 0
  - endpoint = LoRA                           -> ni l'un ni l'autre : à voir sur les ref-
Les deux modèles locaux voient la même question, sans le prompt caché éventuel de l'endpoint : un
déguisement (prompt système, persona) baisse les deux probabilités, mais devrait peu changer le LLR.

Deux modes :
  --telecharger : sur le NŒUD DE LOGIN (internet). Télécharge les modèles candidats dans --models_dir.
  (défaut)      : sur un GPU, via la file de l'équipe. Calcule les probabilités et affiche le bilan.

Candidats par défaut : SmolLM2 135M, 360M et 1.7B Instruct (Hugging Face). Le prompt système par défaut
de leur modèle de conversation (« You are a helpful AI assistant named SmolLM, trained by Hugging
Face ») est celui que les ref- ont recraché. La taille qui donne les probabilités les plus hautes est
la plus probable.

  source slurm/env.sh && python met/score_local.py --telecharger
  bash slurm/soumettre.sh 0 met_local python met/score_local.py \\
      --answers /p/scratch/training2666/huang25/Epi84/model_equality/reference_answers_equipe.jsonl \\
      --key /p/scratch/training2666/huang25/Epi84/model_equality/practice_answers.csv
Sortie : un tableau par endpoint, et results/score_local.csv (une ligne par réponse et par modèle).
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict

CANDIDATS = ["HuggingFaceTB/SmolLM2-135M-Instruct",
             "HuggingFaceTB/SmolLM2-360M-Instruct",
             "HuggingFaceTB/SmolLM2-1.7B-Instruct"]


def dossier_modele(racine, mid):
    return os.path.join(racine, mid.replace("/", "__"))


def court(mid):
    return mid.split("/")[-1].replace("-Instruct", "")


# ----------------------------------------------------------------------------- téléchargement
def telecharger(args):
    os.environ.pop("HF_HUB_OFFLINE", None)          # env.sh le met à 1 pour les nœuds GPU
    from huggingface_hub import snapshot_download
    for mid in args.modeles:
        d = dossier_modele(args.models_dir, mid)
        snapshot_download(mid, local_dir=d,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        print(f"téléchargé : {mid} -> {d}")


# ----------------------------------------------------------------------------- données
def lire_reponses(chemins, prefixe, batterie=None):
    """Réponses d'endpoints. batterie : CSV id,...,prompt (pour les fichiers sans champ prompt, ex. sonde_ref.jsonl)."""
    prompts = {}
    if batterie:
        with open(batterie, newline="", encoding="utf-8") as fh:
            prompts = {r["id"]: r["prompt"] for r in csv.DictReader(fh)}
    items = []
    for ch in chemins:
        with open(ch, encoding="utf-8") as fh:
            for ligne in fh:
                if ligne.strip():
                    r = json.loads(ligne)
                    if str(r["endpoint"]).startswith(prefixe) and (r.get("response") or "").strip():
                        r.setdefault("repeat", r.get("tirage", 0))
                        if "prompt" not in r:
                            r["prompt"] = prompts[r["id"]]
                        items.append(r)
    return items


def lire_cle(chemin):
    cle = {}
    if chemin:
        with open(chemin, newline="", encoding="utf-8-sig") as fh:
            for r in csv.reader(fh):
                if r and not r[0].startswith("endpoint"):
                    cle[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return cle


# ----------------------------------------------------------------------------- modèles
def fp8_simule(model, torch):
    """Quantification fp8 des poids (e4m3, échelle par tenseur), sauf lm_head et embeddings."""
    n = 0
    with torch.no_grad():
        for nom, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and "lm_head" not in nom:
                w = m.weight.data.float()
                s = w.abs().max().clamp(min=1e-12) / 448.0
                m.weight.data = ((w / s).to(torch.float8_e4m3fn).float() * s).to(m.weight.dtype)
                n += 1
    return n


def log_probs(model, tok, items, dev, torch):
    """[(somme des log-probas de la réponse, nombre de tokens)] sachant la question (modèle de conversation)."""
    out = []
    cache = {}
    with torch.no_grad():
        for it in items:
            q = it["prompt"]
            if q not in cache:
                pre = tok.apply_chat_template([{"role": "user", "content": q}],
                                              add_generation_prompt=True, tokenize=False)
                cache[q] = tok(pre, add_special_tokens=False)["input_ids"]
            ids_q = cache[q]
            ids_r = tok(it["response"], add_special_tokens=False)["input_ids"]
            if not ids_r:
                out.append((0.0, 0))
                continue
            ids = torch.tensor([ids_q + ids_r], device=dev)
            lg = model(ids).logits[0, len(ids_q) - 1:-1].float()
            lp = torch.log_softmax(lg, -1).gather(1, ids[0, len(ids_q):, None])[:, 0]
            out.append((float(lp.sum()), len(ids_r)))
    return out


# ----------------------------------------------------------------------------- bilan
def moyenne_et(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), float("inf")
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n)


def mcc(vrai, pred):
    tp = sum(v and p for v, p in zip(vrai, pred)); tn = sum(not v and not p for v, p in zip(vrai, pred))
    fp = sum(not v and p for v, p in zip(vrai, pred)); fn = sum(v and not p for v, p in zip(vrai, pred))
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--telecharger", action="store_true", help="télécharger les modèles (nœud de login)")
    ap.add_argument("--modeles", nargs="+", default=CANDIDATS)
    ap.add_argument("--models_dir", default=os.path.join(os.environ.get("DIR", "."), "models", "met_local"))
    ap.add_argument("--answers", nargs="+", help="fichiers .jsonl de réponses")
    ap.add_argument("--prefix", default="ref-")
    ap.add_argument("--batterie", default=None, help="CSV des prompts si les réponses n'ont pas de champ prompt")
    ap.add_argument("--key", default=None, help="practice_answers.csv (pour les ref-)")
    ap.add_argument("--seuil_z", type=float, default=2.0, help="|z| du LLR au-delà duquel on tranche")
    ap.add_argument("--out", default="results/score_local.csv")
    args = ap.parse_args()

    if args.telecharger:
        telecharger(args)
        return
    if not args.answers:
        sys.exit("--answers manquant")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = lire_reponses(args.answers, args.prefix, args.batterie)
    cle = lire_cle(args.key)
    eps = sorted({it["endpoint"] for it in items})
    print(f"{len(items)} réponses, {len(eps)} endpoints, appareil {dev}")

    res = {}                      # (modèle, variante) -> [(lp, n)] alignée sur items
    for mid in args.modeles:
        d = dossier_modele(args.models_dir, mid)
        if not os.path.isdir(d):
            print(f"absent, ignoré : {d} (lancer --telecharger sur le nœud de login)")
            continue
        tok = AutoTokenizer.from_pretrained(d)
        ex = tok.apply_chat_template([{"role": "user", "content": "QUESTION"}],
                                     add_generation_prompt=True, tokenize=False)
        print(f"\n{court(mid)} : début du modèle de conversation -> {ex[:160]!r}")
        for variante in ("base", "fp8"):
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to(dev).eval()
            if variante == "fp8":
                print(f"  fp8 simulé sur {fp8_simule(model, torch)} couches linéaires")
            res[(mid, variante)] = log_probs(model, tok, items, dev, torch)
            print(f"  {court(mid)} {variante} : {time.time() - t0:.0f} s")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
    if not res:
        sys.exit("aucun modèle chargé")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["endpoint", "id", "repeat", "modele", "variante", "logprob", "ntokens"])
        for (mid, var), v in res.items():
            for it, (lp, n) in zip(items, v):
                w.writerow([it["endpoint"], it["id"], it.get("repeat", 0), court(mid), var, f"{lp:.4f}", n])
    print(f"\nDétail par réponse : {args.out}")

    modeles = sorted({m for m, _ in res}, key=args.modeles.index)
    lab = lambda e: f"{e} {cle[e][1]:>3s} {cle[e][2][:18]:18s}" if e in cle else e

    # 1. log-proba moyenne par token : quelle taille colle le mieux ?
    print("\n1. Log-proba moyenne par token (plus haut = la réponse ressemble plus à ce modèle)")
    print(" " * (len(lab(eps[0])) + 1) + " ".join(f"{court(m)[7:] + '/' + v:>12s}" for m in modeles for v in ("base", "fp8")))
    moy_modele = defaultdict(list)
    for e in eps:
        idx = [i for i, it in enumerate(items) if it["endpoint"] == e]
        cases = []
        for m in modeles:
            for v in ("base", "fp8"):
                lp = sum(res[(m, v)][i][0] for i in idx); n = sum(res[(m, v)][i][1] for i in idx)
                cases.append(f"{lp / max(n, 1):12.3f}")
                if v == "base":
                    moy_modele[m].append(lp / max(n, 1))
        print(lab(e) + " " + " ".join(cases))
    meilleur = max(modeles, key=lambda m: sum(moy_modele[m]) / len(moy_modele[m]))
    print(f"Taille la plus probable : {court(meilleur)}")

    # 2. LLR base - fp8, par endpoint, pour la meilleure taille
    print(f"\n2. LLR par token = log p(base) - log p(fp8), modèle {court(meilleur)}  (z = moyenne / erreur type)")
    verdict = {}
    for e in eps:
        idx = [i for i, it in enumerate(items) if it["endpoint"] == e and res[(meilleur, "base")][i][1] > 0]
        llr = [(res[(meilleur, "base")][i][0] - res[(meilleur, "fp8")][i][0]) / res[(meilleur, "base")][i][1]
               for i in idx]
        m, et = moyenne_et(llr)
        z = m / et if et > 0 else 0.0
        verdict[e] = "base" if z > args.seuil_z else ("fp8" if z < -args.seuil_z else "autre")
        print(f"  {lab(e)}  LLR {m:+.4f} ± {et:.4f}   z {z:+6.2f}   -> {verdict[e]}")

    # 3. regroupement simple et MCC (si clé)
    if cle:
        grp = {e: (verdict[e] if verdict[e] != "autre" else e) for e in eps}
        paires = list(itertools.combinations([e for e in eps if e in cle], 2))
        s = mcc([cle[a][:2] == cle[b][:2] for a, b in paires], [grp[a] == grp[b] for a, b in paires])
        print(f"\n3. Règle simple (base ensemble, fp8 ensemble, « autre » seul) : MCC « même modèle ? » {s:.3f}")
        print("   Attendu sur les ref- : S (01, 03, 05, 06, 08) -> base ; T1 (02) -> fp8 ; T2-T4 (LoRA) -> autre.")


if __name__ == "__main__":
    main()
