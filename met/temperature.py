"""Tâche 2 : vraisemblance des réponses sous le modèle de base à plusieurs températures (repérer les LoRA).

Si un endpoint tire ses tokens de softmax(logits_base / T), alors, pour la bonne T, à chaque position :
    log p_T(token) a pour espérance -H(p_T) et pour variance Var_{p_T}[log p_T]   (exactes, sachant le préfixe).
La somme des écarts sur une réponse est une martingale : z_T = (ll_T - E_T) / sqrt(V_T) suit ~N(0, 1) sous
« même modèle, température T ». Une LoRA tire d'une autre loi : ll_T plus bas que prévu (z_T < 0) à la
température de la famille. Sortie : une ligne par réponse et par T, avec ll_T, E_T, V_T, n.

  bash slurm/soumettre.sh 0 met_temp --venv unlearning env PYTHONPATH=<pylibs_met> python met/temperature.py \\
      --answers <réponses.jsonl> --prefix ep- --batterie met/prompts_ep_phaseA.csv --modeles Qwen/Qwen2.5-3B-Instruct \\
      --endpoints ep-01,ep-02 --out <sortie.csv>
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_local import court, dossier_modele, lire_reponses  # noqa: E402

TEMPS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--prefix", default="ep-")
    ap.add_argument("--batterie", default=None)
    ap.add_argument("--modeles", nargs="+", required=True)
    ap.add_argument("--endpoints", default=None)
    ap.add_argument("--models_dir", default=os.path.join(os.environ.get("DIR", "."), "models", "met_local"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = lire_reponses(args.answers, args.prefix, args.batterie)
    if args.endpoints:
        garde = set(args.endpoints.split(","))
        items = [it for it in items if it["endpoint"] in garde]
    print(f"{len(items)} réponses, appareil {dev}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["endpoint", "id", "repeat", "modele", "T", "ll", "E", "V", "n"])
        for mid in args.modeles:
            d = dossier_modele(args.models_dir, mid); t0 = time.time()
            tok = AutoTokenizer.from_pretrained(d)
            model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).to(dev).eval()
            cache = {}
            with torch.no_grad():
                for it in items:
                    q = it["prompt"]
                    if q not in cache:
                        pre = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
                        cache[q] = tok(pre, add_special_tokens=False)["input_ids"]
                    iq = cache[q]; ir = tok(it["response"], add_special_tokens=False)["input_ids"]
                    if not ir:
                        continue
                    ids = torch.tensor([iq + ir], device=dev)
                    logits = model(ids).logits[0, len(iq) - 1:-1].float()
                    cible = ids[0, len(iq):, None]
                    for T in TEMPS:
                        lg = torch.log_softmax(logits / T, -1)
                        p = lg.exp()
                        ll = lg.gather(1, cible)[:, 0]
                        E = (p * lg).sum(-1)                       # espérance de log p_T(token) = -H(p_T)
                        V = (p * lg * lg).sum(-1) - E * E          # variance de log p_T(token)
                        w.writerow([it["endpoint"], it["id"], it.get("repeat", 0), court(mid), T,
                                    f"{float(ll.sum()):.4f}", f"{float(E.sum()):.4f}", f"{float(V.sum()):.4f}", len(ir)])
            print(f"{court(mid)} : {time.time() - t0:.0f} s", flush=True)
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
    print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
