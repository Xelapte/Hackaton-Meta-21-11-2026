"""Tâche 2 : statistiques token par token des réponses d'un endpoint, sous un modèle local (GPU).

But : séparer les LoRA (poids modifiés) des déguisements (prompt système, persona, tatouage, température),
que le LLR base/fp8 de score_local.py ne distingue pas. Un déguisement agit par le contexte ; une LoRA
change la prédiction du token suivant partout, même loin dans la réponse.

Pour chaque token de chaque réponse, on enregistre (modèle local, sans le prompt caché éventuel) :
  pos      position dans la réponse
  rang     rang du token observé (0 = argmax)
  lp[T]    log softmax(logits / T)[token] pour une grille de températures T
  pit[T]   PIT randomisé : P(X < x) + U * P(X = x) sous softmax(logits / T) (uniforme si c'est le bon modèle)
  ent      entropie à T = 1
Sortie : un .npz (tableaux concaténés + index des réponses), à analyser sur CPU avec met/analyse_logprobs.py.

  bash slurm/soumettre.sh 1 met_lp --venv unlearning env PYTHONPATH=<pylibs_met> python met/logprobs_local.py \\
      --answers <officiel>/model_equality/reference_answers_equipe.jsonl --out results/logprobs_ref.npz
"""
import argparse
import json
import os
import sys

TEMPS = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--prefix", default="ref-")
    ap.add_argument("--modele", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--models_dir", default=os.path.join(os.environ.get("DIR", "."), "models", "met_local"))
    ap.add_argument("--fp8", action="store_true", help="simuler la version fp8 du modèle local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/logprobs_ref.npz")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_local import dossier_modele, fp8_simule, lire_reponses

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = lire_reponses(args.answers, args.prefix)
    d = dossier_modele(args.models_dir, args.modele)
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).to(dev).eval()
    if args.fp8:
        print(f"fp8 simulé sur {fp8_simule(model, torch)} couches")
    T = torch.tensor(TEMPS, device=dev)[:, None, None]

    cols = {k: [] for k in ("pos", "rang", "ent", "lp", "pit")}
    debut, fin = [], []
    n = 0
    cache = {}
    with torch.no_grad():
        for it in items:
            q = it["prompt"]
            if q not in cache:
                pre = tok.apply_chat_template([{"role": "user", "content": q}],
                                              add_generation_prompt=True, tokenize=False)
                cache[q] = tok(pre, add_special_tokens=False)["input_ids"]
            iq = cache[q]
            ir = tok(it["response"], add_special_tokens=False)["input_ids"]
            debut.append(n)
            if ir:
                ids = torch.tensor([iq + ir], device=dev)
                lg = model(ids).logits[0, len(iq) - 1:-1].float()           # (L, V)
                x = ids[0, len(iq):]
                L = len(ir)
                lsT = torch.log_softmax(lg[None] / T, -1)                     # (nT, L, V)
                lpx = lsT.gather(2, x[None, :, None].expand(len(TEMPS), L, 1))[..., 0]
                pT = lsT.exp()
                plus_petit = (pT * (lsT > lpx[..., None])).sum(-1)            # P(X plus probable que x)
                u = torch.rand(L, device=dev)
                pit = 1 - plus_petit - u * lpx.exp()                          # grand = token probable
                ls1 = torch.log_softmax(lg, -1)
                rang = (lg > lg.gather(1, x[:, None])).sum(1)
                ent = -(ls1.exp() * ls1).sum(1)
                cols["pos"].append(np.arange(L)); cols["rang"].append(rang.cpu().numpy())
                cols["ent"].append(ent.cpu().numpy()); cols["lp"].append(lpx.T.cpu().numpy())
                cols["pit"].append(pit.T.cpu().numpy())
                n += L
            fin.append(n)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, temps=np.array(TEMPS), debut=np.array(debut), fin=np.array(fin),
        endpoint=np.array([it["endpoint"] for it in items]), pid=np.array([it["id"] for it in items]),
        repeat=np.array([it.get("repeat", 0) for it in items]),
        **{k: np.concatenate(v) for k, v in cols.items()})
    print(f"{len(items)} réponses, {n} tokens -> {args.out}")


if __name__ == "__main__":
    main()
