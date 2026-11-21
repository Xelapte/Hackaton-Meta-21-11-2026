"""Variantes Q avec transformers : int8, nf4 (bitsandbytes) ou tatouage de Kirchenbauer.

Le papier utilise les implémentations de transformers pour ces trois variantes.
Usage (sur un nœud GPU) :
    python met/generate_hf.py --model <dossier_modele> --prompts met/prompts.jsonl \
        --variant nf4 --n 10 --max_tokens 50 --seed 2 --out results/met/Q_nf4.npz
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import transformers
from packaging import version
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          WatermarkingConfig)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from met.mmd_test import pad  # noqa: E402
from met.generate_vllm import cut_special  # noqa: E402


def load_model(path, variant, dtype):
    kw = {"device_map": "cuda"}
    if variant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif variant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    # « torch_dtype » est devenu « dtype » dans transformers 4.56
    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") \
        else "torch_dtype"
    kw[key] = getattr(torch, dtype)
    return AutoModelForCausalLM.from_pretrained(path, **kw).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--variant", required=True, choices=["int8", "nf4", "watermark",
                                                        "fp16", "fp32"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=50)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"fp16": "float16", "int8": "float16", "nf4": "bfloat16"}.get(args.variant,
                                                                         "float32")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.variant, dtype)
    gen = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
               max_new_tokens=args.max_tokens, num_return_sequences=args.n,
               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
               else tok.eos_token_id)
    if args.variant == "watermark":
        # le papier : biais 2,5 et contexte 1 ; le défaut de transformers est 2,0
        gen["watermarking_config"] = WatermarkingConfig(
            bias=2.5, seeding_scheme="lefthash", context_width=1)

    torch.manual_seed(args.seed)
    special = set(tok.all_special_ids)
    data = {}
    for line in open(args.prompts):
        if not line.strip():
            continue
        p = json.loads(line)
        text = tok.apply_chat_template([{"role": "user", "content": p["text"]}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(ids.to("cuda"), **gen)
        new = out[:, ids.shape[1]:].tolist()
        data[str(p["id"])] = np.array([pad(cut_special(r, special), args.max_tokens)
                                       for r in new])
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"Écrit {args.out} : {len(data)} prompts x {args.n} réponses")


if __name__ == "__main__":
    main()
