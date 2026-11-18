"""Greedy tokens of the q36 parity prompts from a running llama-server (the reference for q36_solo_gate.py's output),
and the comparison of the two.

    python3 tools/ornith/llama_ref.py --url http://127.0.0.1:8081 --out ref.json [--n 64]
    python3 tools/ornith/llama_ref.py --cmp ref.json strata.json
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "qwen35" / "parity_prompts.ids"


def ref(url: str, n: int) -> list:
    out = []
    for line in PROMPTS.read_text().splitlines():
        if not line.strip():
            continue
        ids = [int(x) for x in line.split(",") if x]
        body = {"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False,
                "return_tokens": True, "ignore_eos": True}
        req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        out.append(r["tokens"])
        print(len(out) - 1, repr(r["content"][:80]), flush=True)
    return out


def cmp(a: list, b: list) -> int:
    """The prompts whose first 16 tokens differ (0 = every prompt agrees that far)."""
    bad = 0
    for i, (x, y) in enumerate(zip(a, b)):
        same = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"p{i:02d}: the first {same} of {min(len(x), len(y))} tokens agree")
        bad += same < 16
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cmp", nargs=2)
    a = ap.parse_args()
    if a.cmp:
        r, s = (json.load(open(p)) for p in a.cmp)
        sys.exit(1 if cmp(r if isinstance(r, list) else r["tokens"], s["tokens"]) else 0)
    json.dump(ref(a.url, a.n), open(a.out, "w"))
