#!/usr/bin/env python
"""v57 - Deuxieme tour de fabrication d'images a partir de texte, CIBLE par classe.
Le premier tour (gen_txt.py, 690 noms d'objets) a montre quels objets l'original range dans chaque classe
(ex. "fox figurine" -> classe 0, "copper kettle" -> classe 9). Ici, pour chaque classe, on regenere n images avec ses
3 meilleurs objets et on ne garde que celles que l'original range bien dans cette classe avec une confiance >= seuil.
-> un jeu de controle de remplacement propre et equilibre (images jamais vues, de toutes les classes).
Usage : python gen_txt2.py <classe debut> <classe fin> <n par classe> <graine> <prefixe sortie> [seuil 0.6]"""
import sys, os, time
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30")); sys.path.insert(0, str(B)); sys.path.append(str(B / "pylib_gen"))
import numpy as np, torch, torch.nn.functional as F
c0, c1, n, seed, pref = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; thr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.6
Z, OBJ = [], []
for i in (0, 1):
    m = torch.load(B / f"v57/gen/txt_{i}_meta.pt"); Z.append(m["z"].float()); OBJ.append(m["obj"]); vocab = m["vocab"]
Z = torch.cat(Z); OBJ = torch.cat(OBJ); pred = Z.argmax(1); ok = Z.softmax(1).max(1).values >= 0.5
CTX = ["cinematic photograph, shallow depth of field, soft warm light", "photograph, golden hour, bokeh background", "close-up photo, soft natural light, blurred background",
       "photo taken from above, on a wooden table, warm light", "photograph at dusk, moody lighting", "photo, outdoors, overcast day, muted colors",
       "photograph by a window, morning light, cozy interior", "wide shot photograph, sunset, warm tones", "photo on a shelf, soft light, vintage style",
       "photograph in a garden, morning mist", "photo, winter light, snow in the background", "photograph, plain background, studio light"]
plan = []
for c in range(c0, c1):
    ob = torch.bincount(OBJ[ok & (pred == c)], minlength=len(vocab)).float(); top = ob.topk(3); w = top.values / top.values.sum().clamp_min(1)
    for v, j in zip(w.tolist(), top.indices.tolist()):
        plan += [(c, j)] * int(round(v * n))
prompts = [(c, j, f"a {vocab[j]}, {CTX[(k * 5 + j) % len(CTX)]}") for k, (c, j) in enumerate(plan)]
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(str(B / "ext" / "sd-turbo"), torch_dtype=torch.float16, variant="fp16", local_files_only=True, safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True); g = torch.Generator("cuda").manual_seed(seed); t0 = time.time()
import unlearn as U
from task_template import load_model
orig = load_model(); imgs, ZZ, CC, JJ, tot = [], [], [], [], torch.zeros(100); kept = torch.zeros(100)
for s in range(0, len(prompts), 24):
    ch = prompts[s:s + 24]; im = pipe(prompt=[p for _, _, p in ch], num_inference_steps=2, guidance_scale=0.0, height=512, width=512, output_type="pt", generator=g).images
    x8 = (F.interpolate(im.float(), size=224, mode="bilinear", antialias=True).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu()
    z = U.logits_u8(orig, x8).float().cpu(); cc = torch.tensor([c for c, _, _ in ch]); keep = (z.argmax(1) == cc) & (z.softmax(1).max(1).values >= thr)
    imgs.append(x8[keep]); ZZ.append(z[keep].half()); CC.append(cc[keep]); JJ.append(torch.tensor([j for _, j, _ in ch])[keep]); tot += torch.bincount(cc, minlength=100); kept += torch.bincount(cc[keep], minlength=100)
    if (s // 24) % 40 == 0: print(f"[{(time.time() - t0) / 60:4.1f} min] {s}/{len(prompts)} | gardees {sum(len(i) for i in imgs)}", flush=True)
G = torch.cat(imgs).numpy(); np.save(pref + ".tmp.npy", G); os.rename(pref + ".tmp.npy", pref + ".npy")
torch.save({"z": torch.cat(ZZ), "cls": torch.cat(CC), "obj": torch.cat(JJ), "vocab": vocab, "seuil": thr}, pref + "_meta.pt")
print(f"{len(prompts)} images generees, {len(G)} gardees (bonne classe, confiance >= {thr}) ; {(time.time() - t0) / 60:.1f} min", flush=True)
print("gardees par classe : " + " ".join(f"{c}:{int(kept[c])}/{int(tot[c])}" for c in range(c0, c1)), flush=True); print("fini", flush=True)
