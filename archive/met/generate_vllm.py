"""Génère N réponses par prompt avec vLLM (référence P en fp32, ou variante fp16/bf16).

Usage (sur un nœud GPU, modèle déjà téléchargé) :
    python met/generate_vllm.py --model <dossier_modele> --prompts met/prompts.jsonl \
        --n 10 --max_tokens 50 --dtype float32 --seed 1 --out results/met/P_fp32.npz
prompts.jsonl : une ligne JSON par prompt, {"id": "...", "text": "..."}.
Sortie : un .npz {id_du_prompt: tableau (n, max_tokens) d'ids de tokens, -1 = bourrage}.
IMPORTANT : une graine différente pour chaque fichier (P et Q doivent être indépendants).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from met.mmd_test import pad  # noqa: E402


def cut_special(ids, special):
    """Coupe la réponse au premier token spécial (fin de tour, EOS…)."""
    for i, t in enumerate(ids):
        if t in special:
            return list(ids[:i])
    return list(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=10, help="réponses par prompt")
    ap.add_argument("--max_tokens", type=int, default=50)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16",
                                                          "float16"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [json.loads(line) for line in open(args.prompts) if line.strip()]
    # le gabarit de chat produit déjà le token de début : pas de second BOS
    token_prompts = [{"prompt_token_ids": tok(
        tok.apply_chat_template([{"role": "user", "content": p["text"]}],
                                tokenize=False, add_generation_prompt=True),
        add_special_tokens=False)["input_ids"]} for p in prompts]
    print("Longueur du 1er prompt en tokens :", len(token_prompts[0]["prompt_token_ids"]))

    llm = LLM(model=args.model, dtype=args.dtype, seed=args.seed,
              gpu_memory_utilization=args.gpu_mem)
    # température 1, sans top-p : on fixe tout explicitement (fiche 2)
    params = SamplingParams(n=args.n, temperature=1.0, top_p=1.0,
                            max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(token_prompts, params)

    special = set(tok.all_special_ids)
    data = {str(p["id"]): np.array([pad(cut_special(c.token_ids, special),
                                         args.max_tokens) for c in o.outputs])
            for p, o in zip(prompts, outs)}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"Écrit {args.out} : {len(data)} prompts x {args.n} réponses")


if __name__ == "__main__":
    main()
