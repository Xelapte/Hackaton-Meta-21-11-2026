"""Compare deux échantillons de réponses avec le test MMD-Hamming (fiche 2).

Entrées acceptées :
  - .npz produits par generate_vllm.py / generate_hf.py (ids de tokens) ;
  - .json {id_du_prompt: [texte, texte, …]} (réponses d'API), comparées en caractères.
Usage :
    python met/compare.py results/met/P_fp32.npz results/met/Q_nf4.npz --name nf4
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.scores import log_score  # noqa: E402
from met.mmd_test import mmd_test, pad  # noqa: E402


def load(path, max_chars):
    if path.endswith(".npz"):
        return {k: v for k, v in np.load(path).items()}
    raw = json.load(open(path))
    return {k: np.array([pad([ord(c) for c in t], max_chars) for t in texts])
            for k, texts in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p_file")
    ap.add_argument("q_file")
    ap.add_argument("--B", type=int, default=1000, help="nombre de permutations")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max_chars", type=int, default=1000,
                    help="longueur en caractères pour les fichiers .json")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    DP, DQ = load(args.p_file, args.max_chars), load(args.q_file, args.max_chars)
    common = sorted(set(DP) & set(DQ))
    if not common:
        sys.exit("Aucun prompt commun entre les deux fichiers")
    if any(DP[k].shape[1] != DQ[k].shape[1] for k in common):
        sys.exit("Longueurs différentes : comparez des fichiers générés avec le même L")
    mmd, p, _ = mmd_test({k: DP[k] for k in common}, {k: DQ[k] for k in common},
                         B=args.B)
    verdict = "DIFFÉRENTES" if p < args.alpha else "rien à signaler"
    print(f"{len(common)} prompts | MMD = {mmd:.5f} | p-value = {p:.4f} -> {verdict}")
    log_score("met", {"P": args.p_file, "Q": args.q_file, "B": args.B,
                      "name": args.name}, round(p, 6),
              notes=f"mmd={mmd:.5f} {verdict}")


if __name__ == "__main__":
    main()
