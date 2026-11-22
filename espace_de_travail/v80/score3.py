"""v80 : test 'comptes agreges' sous le modele de base local (CPU).
Pour chaque endpoint : O[v] = nb de fois ou le token v a ete tire, E[v] = somme des p_t(v) (attendu exact, sachant chaque prefixe),
W[v] = somme des p_t(v)(1 - p_t(v)) (variance exacte). Sous 'meme modele, T = 0.7', O - E est une martingale centree.
Un petit decalage coherent des logits (LoRA faible, quantification) s'additionne sur toutes les positions (effet du 1er ordre).
Sortie : un .pt {endpoint: {"O","E","W","ll","Ez","Vz","n","nrep", "par_id": {...}}}.
Usage : python score3.py <dossier_modele> <ep-..,ep-..> <sortie.pt> <fichier.jsonl> [...] [--batterie csv]"""
import csv, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(int(os.environ.get("NT", "8")))
a = sys.argv[1:]; bat = None
if "--batterie" in a:
    i = a.index("--batterie"); bat = a[i + 1]; a = a[:i] + a[i + 2:]
d, eps, out, files = a[0], set(a[1].split(",")), a[2], a[3:]
T0 = float(os.environ.get("T0", "0.7"))
prompts = {}
if bat:
    prompts = {r["id"]: r["prompt"] for r in csv.DictReader(open(bat, newline="", encoding="utf-8"))}
items = []
for f in files:
    for l in open(f, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["endpoint"] in eps and (r.get("response") or "").strip():
                if "prompt" not in r:
                    r["prompt"] = prompts[r["id"]]
                items.append(r)
print(len(items), "reponses", flush=True)
tok = AutoTokenizer.from_pretrained(d)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16).to(dev).eval()
Vn = model.get_output_embeddings().weight.shape[0]
res = {}
def acc(ep, pid):
    e = res.setdefault(ep, {"par_id": {}})
    return e["par_id"].setdefault(pid, {"O": torch.zeros(Vn, dtype=torch.float64), "E": torch.zeros(Vn, dtype=torch.float64),
                                       "W": torch.zeros(Vn, dtype=torch.float64), "z": [], "n": 0, "first": []})
cache = {}; t0 = time.time()
with torch.no_grad():
    for k, it in enumerate(items):
        q = it["prompt"]
        if q not in cache:
            pre = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
            cache[q] = tok(pre, add_special_tokens=False)["input_ids"]
        iq = cache[q]; ir = tok(it["response"], add_special_tokens=False)["input_ids"]
        if not ir:
            continue
        ids = torch.tensor([iq + ir], device=dev)
        logits = model(ids).logits[0, len(iq) - 1:-1].double().cpu()
        lg = torch.log_softmax(logits / T0, -1); p = lg.exp(); cible = torch.tensor(ir)
        A = acc(it["endpoint"], it["id"])
        A["O"].index_add_(0, cible, torch.ones(len(ir), dtype=torch.float64)); A["E"] += p.sum(0); A["W"] += (p * (1 - p)).sum(0)
        ll = lg.gather(1, cible[:, None])[:, 0]; E = (p * lg).sum(-1); Vv = (p * lg * lg).sum(-1) - E * E
        A["z"].append((float(ll.sum()), float(E.sum()), float(Vv.sum()), len(ir))); A["n"] += len(ir)
        A["first"].append((int(cible[0]), float(p[0, cible[0]])))
        if k % 20 == 0:
            print(k, round(time.time() - t0), "s", flush=True)
torch.save(res, out)
print("fini", round(time.time() - t0), "s ->", out)
