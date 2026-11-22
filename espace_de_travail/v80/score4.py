"""v80 : pour les prompts 'choix aleatoires' : sauvegarde, par reponse, les tokens tires et les probabilites exactes (modele de base
local, T0) des tokens 'suivis' (tous les tokens distincts vus dans les reponses a ce prompt, tous endpoints confondus).
Permet ensuite des tests du 1er ordre hors ligne (exces par token, exces conditionnel au token precedent).
Usage : python score4.py <dossier_modele> <ep-..,ep-..> <sortie.pt> <fichier.jsonl> [...]   (env T0, defaut 0.7)"""
import json, os, sys, time, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(int(os.environ.get("NT", "8")))
d, eps, out, files = sys.argv[1], set(sys.argv[2].split(",")), sys.argv[3], sys.argv[4:]
T0 = float(os.environ.get("T0", "0.7"))
items = []
for f in files:
    for l in open(f, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["endpoint"] in eps and (r.get("response") or "").strip():
                items.append(r)
tok = AutoTokenizer.from_pretrained(d)
for it in items:
    it["ir"] = tok(it["response"], add_special_tokens=False)["input_ids"]
suivis = collections.defaultdict(set)
for it in items:
    suivis[it["id"]].update(it["ir"])
suivis = {k: torch.tensor(sorted(v)) for k, v in suivis.items()}
print(len(items), "reponses ; tokens suivis :", {k: len(v) for k, v in suivis.items()}, flush=True)
model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32).eval()
res = {"suivis": suivis, "rep": []}; cache = {}; t0 = time.time()
with torch.no_grad():
    for k, it in enumerate(items):
        q = it["prompt"]
        if q not in cache:
            pre = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
            cache[q] = tok(pre, add_special_tokens=False)["input_ids"]
        iq, ir = cache[q], it["ir"]
        if not ir:
            continue
        ids = torch.tensor([iq + ir])
        logits = model(ids).logits[0, len(iq) - 1:-1].double()
        p = torch.softmax(logits / T0, -1)
        res["rep"].append({"endpoint": it["endpoint"], "id": it["id"], "repeat": it["repeat"], "x": torch.tensor(ir),
                           "P": p[:, suivis[it["id"]]].float(), "px": p.gather(1, torch.tensor(ir)[:, None])[:, 0].float()})
        if k % 40 == 0:
            print(k, round(time.time() - t0), "s", flush=True)
torch.save(res, out)
print("fini", round(time.time() - t0), "s ->", out)
