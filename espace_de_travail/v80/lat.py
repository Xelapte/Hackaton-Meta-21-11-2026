import os, time, requests, sys
from concurrent.futures import ThreadPoolExecutor
K = open(os.path.expanduser("~/.hack_api_key")).read().strip()
B = "http://34.147.229.55/33-model-equality/v1"
s = requests.Session(); s.headers["X-API-Key"] = K
def one(a):
    ep, mt = a; t = time.time()
    r = s.post(B + "/chat/completions", timeout=180, json={"model": ep, "max_tokens": mt, "messages": [{"role": "user", "content": "Write a long, detailed story about an old lighthouse keeper and a storm."}]})
    c = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else r.text[:100]
    return ep, mt, r.status_code, round(time.time() - t, 1), len(c), r.json()["choices"][0].get("finish_reason") if r.status_code == 200 else None
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    for x in ex.map(one, [("ref-01", 256), ("ref-02", 256), ("ref-03", 32), ("ref-04", 32)]): print(x)
print("total", round(time.time() - t0, 1))
