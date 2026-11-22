#!/usr/bin/env python
"""v55 - Variantes "fraiches" des images a oublier, fabriquees par un modele de diffusion (SD-Turbo, image vers image).
Chaque variante garde le style et l'objet de l'image de depart mais redessine la scene : c'est une image que le modele n'a
JAMAIS vue, du meme type que le jeu de donnees. On mesure aussi ce qu'en pense l'original (bonne classe ? confiance ?).
Usage : python gen_var.py <debut> <fin> <force 0-1> <graine> <sortie.npy> [planche.jpg]"""
import sys, os, time
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
B = Path("/p/scratch/training2666/huang25/Epi84/unlearning"); sys.path.insert(0, str(B / "v30")); sys.path.insert(0, str(B)); sys.path.append(str(B / "pylib_gen"))
import numpy as np, torch, torch.nn.functional as F
a, b, strength, seed, out = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sheet = sys.argv[6] if len(sys.argv) > 6 else None
from diffusers import StableDiffusionImg2ImgPipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(B / "ext" / "sd-turbo"), torch_dtype=torch.float16, variant="fp16", local_files_only=True, safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True)
forget = np.load(B / "forget_images.npz")["images"]; t0 = time.time(); outs = []
PROMPT = "cinematic photograph, shallow depth of field, soft warm light"; steps = 4
g = torch.Generator("cuda").manual_seed(seed)
for s in range(a, b, 20):
    x = torch.from_numpy(forget[s:min(s + 20, b)]).permute(0, 3, 1, 2).float().div(255).cuda()
    x5 = F.interpolate(x, size=512, mode="bicubic", align_corners=False).clamp(0, 1)
    im = pipe(prompt=[PROMPT] * len(x), image=x5, strength=strength, num_inference_steps=steps, guidance_scale=0.0, output_type="pt", generator=g).images
    y = F.interpolate(im.float(), size=224, mode="bilinear", antialias=True).clamp(0, 1)
    outs.append((y * 255).round().byte().permute(0, 2, 3, 1).cpu())
    if (s - a) % 400 == 0: print(f"[{(time.time() - t0) / 60:4.1f} min] {s - a}/{b - a}", flush=True)
G = torch.cat(outs).numpy(); np.save(out + ".tmp.npy", G); os.rename(out + ".tmp.npy", out); print("saved", out, G.shape, f"{(time.time() - t0) / 60:.1f} min", flush=True)
del pipe; torch.cuda.empty_cache()
# ce qu'en pense l'original
import unlearn as U
from task_template import load_model
orig = load_model(); lo = U.logits_u8(orig, torch.from_numpy(forget[a:b])).float(); y_ = lo.argmax(1); zg = U.logits_u8(orig, torch.from_numpy(G)).float()
p2 = F.softmax(zg / 2, 1)[torch.arange(len(y_), device=zg.device), y_]; L = F.cross_entropy(zg, y_, reduction="none")
torch.save({"z": zg.half().cpu(), "y": y_.cpu(), "a": a, "b": b, "force": strength, "graine": seed}, out.replace(".npy", "_logits.pt"))
print(f"force {strength} : l'original reconnait la bonne classe sur {(zg.argmax(1) == y_).float().mean():.3f} des variantes | loss {L.mean():.2f} | proba T=2 de la vraie classe {p2.mean():.2f} (image exacte : {F.softmax(lo / 2, 1).max(1).values.mean():.2f})", flush=True)
if sheet:
    from PIL import Image
    k = min(10, len(G)); idx = np.linspace(0, len(G) - 1, k).astype(int)
    row = lambda arr: np.concatenate([np.asarray(Image.fromarray(arr[i]).resize((160, 160))) for i in idx], 1)
    Image.fromarray(np.concatenate([row(forget[a:b]), row(G)], 0)).save(sheet, quality=88)
print("fini", flush=True)
