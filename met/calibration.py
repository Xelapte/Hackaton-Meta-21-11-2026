"""Tâche 2 : test de calibration sous le modèle de base local (pour repérer les LoRA).

Si un endpoint échantillonne le modèle de base, chaque token de sa réponse est tiré de p_base(·|contexte) (à une
température près) : en moyenne, log p_base(token) ≈ -H(p_base(·|contexte)) à température 1, et plus haut si
la température est < 1. Une LoRA tire ses tokens d'une autre loi q : log p_base(token) baisse sans que l'entropie
de p_base baisse autant (entropie croisée > entropie). Statistiques par réponse, sous le modèle local :
  lp    : somme des log p_base(token)       H   : somme des entropies de p_base aux mêmes positions
  top1  : nombre de tokens égaux à l'argmax  n   : nombre de tokens
Analyse (analyse_calibration.py) : calib = (lp + H) / n, top1 / n, par endpoint, comparés au reste de la famille.

  bash slurm/soumettre.sh 0 met_calib --venv unlearning env PYTHONPATH=<pylibs_met> python met/calibration.py \\
      --answers <réponses.jsonl> --prefix ep- --batterie met/prompts_ep_phaseA.csv --modeles Qwen/Qwen2.5-3B-Instruct \\
      --out <sortie.csv>
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_local import court, dossier_modele, lire_reponses  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--prefix", default="ep-")
    ap.add_argument("--batterie", default=None)
    ap.add_argument("--modeles", nargs="+", required=True)
    ap.add_argument("--endpoints", default=None, help="liste séparée par des virgules (défaut : tous)")
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
        w.writerow(["endpoint", "id", "repeat", "modele", "lp", "H", "top1", "n"])
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
                    lg = torch.log_softmax(model(ids).logits[0, len(iq) - 1:-1].float(), -1)
                    cible = ids[0, len(iq):]
                    lp = lg.gather(1, cible[:, None])[:, 0]
                    H = -(lg.exp() * lg).sum(-1)
                    top1 = (lg.argmax(-1) == cible).sum()
                    w.writerow([it["endpoint"], it["id"], it.get("repeat", 0), court(mid), f"{float(lp.sum()):.4f}",
                                f"{float(H.sum()):.4f}", int(top1), len(ir)])
            print(f"{court(mid)} : {time.time() - t0:.0f} s", flush=True)
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
    print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
