import os, json, time, requests
K = open(os.path.expanduser("~/.hack_api_key")).read().strip()
B = "http://34.147.229.55/33-model-equality/v1"
s = requests.Session(); s.headers["X-API-Key"] = K
t = time.time()
r = s.post(B + "/chat/completions", timeout=180, json={"model": "ref-01", "max_tokens": 256, "logprobs": True, "top_logprobs": 5,
           "messages": [{"role": "user", "content": "Write a long story about a forest."}]})
print("HTTP", r.status_code, round(time.time() - t, 2), "s")
b = r.json()
def shape(o, d=0):
    if isinstance(o, dict):
        return {k: shape(v, d + 1) for k, v in o.items()}
    if isinstance(o, list):
        return [shape(o[0], d + 1)] if o else []
    return (o if not isinstance(o, str) or len(o) < 80 else o[:80] + "...")
print(json.dumps(shape(b), indent=1)[:1800])
m = s.get(B + "/models", timeout=60).json()["data"]
print({x["id"]: x["budget"]["limit"] - x["budget"]["used"] for x in m if x["id"].startswith("ep-")})
print([ (k, (v if not isinstance(v,(dict,list)) else '...')) for k, v in m[0].items()])
