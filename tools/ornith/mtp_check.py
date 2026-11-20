"""Sampled MTP drafting check: seeded, sampled requests; prints one hash per answer and the decode speed.

Run it once with drafting under sampling (the default) and once with STRATA_Q36_SAMPLED_MTP=0: the hashes must match
(same draws in the same order on bitwise-equal rows), and a second run with the same mode must match too.

    API_KEY=... python3 -I mtp_check.py http://127.0.0.1:8000 > run-a.txt
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number modulo 1e9+7 in O(log n). Explain briefly.",
    "Explain how a hash map handles collisions, with a short C example.",
    "Given an array of integers, find the length of the longest strictly increasing subsequence. Give an O(n log n) solution in C++.",
    "What is the difference between a process and a thread? Answer in five sentences.",
]
SEEDS = [11, 22, 33]


def ask(url, key, prompt, seed):
    body = {"model": "ornith", "messages": [{"role": "user", "content": prompt}], "temperature": 0.6, "top_p": 0.95,
            "top_k": 20, "seed": seed, "max_tokens": 600}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    return text, d["usage"]["completion_tokens"], time.time() - t0


def main():
    url, key = sys.argv[1], os.environ["API_KEY"]
    tok = sec = 0
    for p in range(len(PROMPTS)):
        for s in SEEDS:
            text, n, t = ask(url, key, PROMPTS[p], s)
            tok, sec = tok + n, sec + t
            print(f"prompt {p} seed {s}: {hashlib.sha256(text.encode()).hexdigest()[:16]} {n} tokens {n / t:.1f} tok/s", flush=True)
    print(f"all: {tok} tokens in {sec:.1f} s = {tok / sec:.1f} tok/s (prompt time included)")


if __name__ == "__main__":
    main()
