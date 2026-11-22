"""v80 : vraisemblance exacte des nouvelles reponses (collecte2.jsonl) sous le modele de base local de la famille (CPU, leger).
Meme calcul que met/temperature.py de l'equipe (z = (ll - E)/sqrt(V) sous softmax(logits/T)), mais incremental et sur CPU.
Usage : python score2.py <dossier_modele> <ep-13,ep-19> <sortie.csv>"""
import csv, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(8)
V = "/p/scratch/training2666/huang25/Epi84/unlearning/v80/"
d, eps, out = sys.argv[1], set(sys.argv[2].split(",")), sys.argv[3]
TEMPS = [0.6, 0.65, 0.7, 0.75, 0.8]
done = set()
if os.path.exists(out):
    done = {(r["endpoint"], r["id"], r["repeat"]) for r in csv.DictReader(open(out))}
items = [r for r in (json.loads(l) for l in open(V + "collecte2.jsonl") if l.strip())
         if r["endpoint"] in eps and (r["endpoint"], r["id"], str(r["repeat"])) not in done and (r.get("response") or "").strip()]
print(len(items), "reponses a noter,", len(done), "deja faites", flush=True)
if not items:
    sys.exit(0)
tok = AutoTokenizer.from_pretrained(d)
model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32).eval()
new = not os.path.exists(out)
fh = open(out, "a", newline=""); w = csv.writer(fh)
if new:
    w.writerow(["endpoint", "id", "repeat", "n", "T", "ll", "E", "V"])
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
        ids = torch.tensor([iq + ir])
        logits = model(ids).logits[0, len(iq) - 1:-1].double()
        cible = ids[0, len(iq):, None]
        for T in TEMPS:
            lg = torch.log_softmax(logits / T, -1); p = lg.exp()
            ll = lg.gather(1, cible)[:, 0]; E = (p * lg).sum(-1); Vv = (p * lg * lg).sum(-1) - E * E
            w.writerow([it["endpoint"], it["id"], it["repeat"], len(ir), T, f"{float(ll.sum()):.5f}", f"{float(E.sum()):.5f}", f"{float(Vv.sum()):.5f}"])
        fh.flush()
        if k % 10 == 0:
            print(k, round(time.time() - t0), "s", flush=True)
print("fini", round(time.time() - t0), "s")
