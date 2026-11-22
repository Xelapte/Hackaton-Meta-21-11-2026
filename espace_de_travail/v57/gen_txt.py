#!/usr/bin/env python
"""v57 - Images FRAICHES fabriquees a partir de TEXTE (SD-Turbo), pour obtenir des images de TOUTES les classes du modele,
y compris celles dont on n'a aucune image (55 classes sur 100 sont absentes des images a oublier).
1. Pour chaque nom d'objet d'une liste large (vocab.txt), on genere n images dans le style du jeu de donnees.
2. L'original les classe. Une image ou il est sur de lui est une image "de sa distribution" : on la garde, avec sa reponse.
   -> jeu de controle de remplacement (mesure locale du retain) + ancre pour l'entrainement.
Usage : python gen_txt.py <vocab.txt> <debut> <fin> <n_par_objet> <graine> <prefixe_sortie> [seuil de confiance, 0.5]"""
import sys, os, time
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30")); sys.path.insert(0, str(B)); sys.path.append(str(B / "pylib_gen"))
import numpy as np, torch, torch.nn.functional as F
vocab = [l.strip() for l in open(sys.argv[1]) if l.strip()]; a, b, n, seed, pref = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
thr = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5; b = min(b, len(vocab))
CTX = ["cinematic photograph, shallow depth of field, soft warm light", "photograph, golden hour, bokeh background", "close-up photo, soft natural light, blurred background",
       "photo taken from above, on a wooden table, warm light", "photograph at dusk, moody lighting", "photo, outdoors, overcast day, muted colors",
       "photograph by a window, morning light, cozy interior", "wide shot photograph, sunset, warm tones"]
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(str(B / "ext" / "sd-turbo"), torch_dtype=torch.float16, variant="fp16", local_files_only=True, safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True); g = torch.Generator("cuda").manual_seed(seed); t0 = time.time()
import unlearn as U
from task_template import load_model
orig = load_model()
prompts = [(i, f"a {vocab[i]}, {CTX[(k + i) % len(CTX)]}") for i in range(a, b) for k in range(n)]
imgs, Z, OBJ = [], [], []
for s in range(0, len(prompts), 24):
    chunk = prompts[s:s + 24]
    im = pipe(prompt=[p for _, p in chunk], num_inference_steps=2, guidance_scale=0.0, height=512, width=512, output_type="pt", generator=g).images
    x8 = (F.interpolate(im.float(), size=224, mode="bilinear", antialias=True).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu()
    z = U.logits_u8(orig, x8).float().cpu(); keep = z.softmax(1).max(1).values >= thr
    imgs.append(x8[keep]); Z.append(z.half()); OBJ += [i for i, _ in chunk]
    if (s // 24) % 40 == 0: print(f"[{(time.time() - t0) / 60:4.1f} min] {s}/{len(prompts)} | gardees {sum(len(i) for i in imgs)}", flush=True)
Z = torch.cat(Z); OBJ = torch.tensor(OBJ); conf = Z.float().softmax(1).max(1).values; keep = conf >= thr
G = torch.cat(imgs).numpy(); np.save(pref + ".tmp.npy", G); os.rename(pref + ".tmp.npy", pref + ".npy")
torch.save({"z": Z, "obj": OBJ, "keep": keep, "vocab": vocab, "a": a, "b": b, "n": n, "graine": seed, "seuil": thr}, pref + "_meta.pt")
pred = Z.float().argmax(1)
print(f"{len(Z)} images generees, {int(keep.sum())} gardees (confiance >= {thr}) ; classes couvertes par les gardees : {pred[keep].unique().numel()}/100 ; {(time.time() - t0) / 60:.1f} min", flush=True)
for i in range(a, b):
    m = OBJ == i; c = pred[m].mode().values.item(); print(f"  {vocab[i]:32s} -> classe {c:2d} sur {(pred[m] == c).float().mean():.2f} des images | confiance moyenne {conf[m].mean():.2f}", flush=True)
print("fini", flush=True)
