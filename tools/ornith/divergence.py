"""Where Strata's greedy tokens leave llama.cpp's, how likely llama.cpp found each side's token there (teacher-forced on
Strata's prefix).  A near tie means ordinary numeric drift; Strata's token at ~0 means an error.

    python3 tools/ornith/divergence.py --url http://127.0.0.1:8081 ref.json strata.json
"""
import argparse
import json
import math
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "qwen35" / "parity_prompts.ids"

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("ref")
ap.add_argument("strata")
a = ap.parse_args()
ids = [[int(x) for x in l.split(",") if x] for l in PROMPTS.read_text().splitlines() if l.strip()]
ref = json.load(open(a.ref))
got = json.load(open(a.strata))["tokens"]
for i, (r, s) in enumerate(zip(ref, got)):
    k = next((j for j, (x, y) in enumerate(zip(r, s)) if x != y), None)
    if k is None:
        print(f"p{i:02d}: no divergence")
        continue
    body = {"prompt": ids[i] + s[:k], "n_predict": 1, "temperature": 0, "n_probs": 10, "cache_prompt": False}
    req = urllib.request.Request(a.url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    top = {t["id"]: math.exp(t["logprob"]) for t in json.load(urllib.request.urlopen(req))["completion_probabilities"][0]["top_logprobs"]}
    print(f"p{i:02d} step {k}: llama.cpp gives {top.get(r[k], 0):.3f} to its token, {top.get(s[k], 0):.3f} to Strata's")
