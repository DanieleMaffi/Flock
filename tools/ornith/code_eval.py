"""A small coding-quality comparison: LiveCodeBench problems through an OpenAI-compatible server, scored by their tests.

    # 1. a fixed sample of problems, their tests decoded once into plain JSON
    python3 -I code_eval.py select --data test6.jsonl --out tasks.jsonl --easy 8 --medium 20 --hard 12 --seed 1
    # 2. one answer per problem from a server (resumable: problems already in --out are skipped)
    python3 -I code_eval.py gen --tasks tasks.jsonl --url http://127.0.0.1:8000 --key-env API_KEY --out gen-flash.jsonl
    # 3. the answers against every test, each program in a child process with time and memory limits.  Run this step
    #    in a container without network (docker run --network none ...): the programs are model-written code.
    python3 -I code_eval.py score --tasks tasks.jsonl --gen gen-flash.jsonl

The prompts are LiveCodeBench's (lcb_runner, code generation).  A problem counts as solved when its program passes every
public and private test; the time limit per test is 6 s, as there.
"""
import argparse
import base64
import io
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
          "correct Python program that matches the specification and passes all tests.")
FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample "
             "inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
             "inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
FMT_CALL = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"


# ---------------------------------------------------------------------------------------------------------- select
class _PlainData(pickle.Unpickler):
    """The private tests are pickled; only plain data may come out (no class is ever looked up)."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused {module}.{name}: the tests are plain data")


def _private_tests(blob: str) -> list:
    try:
        return json.loads(blob)
    except ValueError:
        raw = zlib.decompress(base64.b64decode(blob.encode()))
        return json.loads(_PlainData(io.BytesIO(raw)).load())


def select(a):
    rows = [json.loads(l) for l in open(a.data)]
    rng = random.Random(a.seed)
    out = []
    for diff, n in (("easy", a.easy), ("medium", a.medium), ("hard", a.hard)):
        pool = sorted((r for r in rows if r["difficulty"] == diff), key=lambda r: r["question_id"])
        out += rng.sample(pool, min(n, len(pool)))
    with open(a.out, "w") as f:
        for r in out:
            meta = json.loads(r["metadata"]) if r.get("metadata") else {}
            tests = json.loads(r["public_test_cases"]) + _private_tests(r["private_test_cases"])
            f.write(json.dumps({"id": r["question_id"], "title": r["question_title"], "difficulty": r["difficulty"],
                                "platform": r["platform"], "question": r["question_content"],
                                "starter": r.get("starter_code") or "", "func": meta.get("func_name"),
                                "tests": [{"input": t["input"], "output": t["output"], "type": t["testtype"]} for t in tests]})
                    + "\n")
    print(f"{len(out)} problems, {sum(len(json.loads(l)['tests']) for l in open(a.out))} tests -> {a.out}")


# ------------------------------------------------------------------------------------------------------------- gen
def prompt(t: dict) -> str:
    fmt = (FMT_CALL + f"```python\n{t['starter']}\n```\n\n") if t["starter"] else FMT_STDIN
    return f"### Question:\n{t['question']}\n\n### Format: {fmt}### Answer: (use the provided format with backticks)\n\n"


def gen(a):
    tasks = [json.loads(l) for l in open(a.tasks)]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    key = os.environ.get(a.key_env, "") if a.key_env else ""
    for t in tasks:
        if t["id"] in done:
            continue
        body = {"model": a.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(t)}],
                "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=a.timeout))
            msg = r["choices"][0]["message"]
            rec = {"id": t["id"], "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
                   "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage"), "seconds": time.time() - t0}
        except Exception as e:   # noqa: BLE001 - recorded as a failed answer, the run goes on
            rec = {"id": t["id"], "content": "", "error": str(e)[:300], "seconds": time.time() - t0}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        u = rec.get("usage") or {}
        print(f"{t['id']} {t['difficulty']}: {u.get('completion_tokens', 0)} tokens in {rec['seconds']:.0f} s "
              f"({rec.get('finish') or rec.get('error', '')})", flush=True)


# ----------------------------------------------------------------------------------------------------------- score
def extract(content: str) -> str:
    """The last ```python block of the answer (after any reasoning), else the last ``` block, else nothing."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    blocks = re.findall(r"```python\s*\n(.*?)```", content, re.S) or re.findall(r"```\s*\n(.*?)```", content, re.S)
    return blocks[-1] if blocks else ""


RUNNER_CALL = r'''
import json, sys
sys.setrecursionlimit(1000000)
src = open(sys.argv[1]).read()
func = sys.argv[2]
args = [json.loads(line) for line in open(sys.argv[3]).read().split("\n") if line.strip()]
ns = {}
exec("from typing import *\nfrom collections import *\nfrom math import *\nfrom heapq import *\nfrom bisect import *\nfrom itertools import *\nfrom functools import *\n" + src, ns)
res = getattr(ns["Solution"](), func)(*args)
print(json.dumps(res))
'''


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def _same(got: str, want: str, functional: bool) -> bool:
    if functional:
        try:
            g, w = json.loads(got), json.loads(want)
        except ValueError:
            return got.strip() == want.strip()
        if isinstance(g, float) or isinstance(w, float):
            try:
                return abs(float(g) - float(w)) <= 1e-6 * max(1.0, abs(float(w)))
            except (TypeError, ValueError):
                return False
        return g == w
    gl, wl = got.strip().split(), want.strip().split()
    if gl == wl:
        return True
    if len(gl) != len(wl):
        return False
    for x, y in zip(gl, wl):
        if x != y:
            try:
                if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                    return False
            except ValueError:
                return False
    return True


def run_task(t: dict, code: str, per_test: float) -> tuple:
    """(passed, total, first failure reason)."""
    if not code.strip():
        return 0, len(t["tests"]), "no code"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "sol.py")
        open(src, "w").write(code)
        runner = os.path.join(d, "run_call.py")
        open(runner, "w").write(RUNNER_CALL)
        passed = 0
        for i, tc in enumerate(t["tests"]):
            functional = tc["type"] == "functional"
            if functional:
                inp = os.path.join(d, "args.txt")
                open(inp, "w").write(tc["input"])
                cmd, stdin = [sys.executable, "-I", runner, src, t["func"] or "", inp], ""
            else:
                cmd, stdin = [sys.executable, "-I", src], tc["input"]
            try:
                p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=per_test, cwd=d,
                                   preexec_fn=_limits)
            except subprocess.TimeoutExpired:
                return passed, len(t["tests"]), f"test {i}: time limit"
            if p.returncode != 0:
                return passed, len(t["tests"]), f"test {i}: error {p.stderr.strip().splitlines()[-1][:120] if p.stderr.strip() else p.returncode}"
            if not _same(p.stdout, tc["output"], functional):
                return passed, len(t["tests"]), f"test {i}: wrong answer"
            passed += 1
    return passed, len(t["tests"]), ""


def score(a):
    tasks = {json.loads(l)["id"]: json.loads(l) for l in open(a.tasks)}
    answers = {}
    for l in open(a.gen):
        r = json.loads(l)
        answers[r["id"]] = r
    by = {}
    rows = []
    for tid, t in tasks.items():
        r = answers.get(tid)
        code = extract(r["content"]) if r else ""
        p, n, why = run_task(t, code, a.per_test)
        ok = p == n and n > 0
        by.setdefault(t["difficulty"], [0, 0])
        by[t["difficulty"]][0] += ok
        by[t["difficulty"]][1] += 1
        toks = ((r or {}).get("usage") or {}).get("completion_tokens", 0)
        rows.append({"id": tid, "difficulty": t["difficulty"], "solved": ok, "passed": p, "tests": n, "why": why,
                     "tokens": toks, "finish": (r or {}).get("finish")})
        print(f"{tid:12s} {t['difficulty']:6s} {'SOLVED' if ok else 'failed':6s} {p}/{n} {why}", flush=True)
    solved = sum(v[0] for v in by.values())
    print(f"solved {solved} of {len(tasks)} ({100 * solved / max(1, len(tasks)):.1f}%): " +
          ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by.items())))
    if a.out:
        json.dump({"solved": solved, "total": len(tasks), "by": by, "rows": rows}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--easy", type=int, default=8)
    s.add_argument("--medium", type=int, default=20)
    s.add_argument("--hard", type=int, default=12)
    s.add_argument("--seed", type=int, default=1)
    g = sub.add_parser("gen")
    g.add_argument("--tasks", required=True)
    g.add_argument("--url", required=True)
    g.add_argument("--key-env", default="")
    g.add_argument("--model", default="model")
    g.add_argument("--out", required=True)
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--timeout", type=float, default=3600)
    c = sub.add_parser("score")
    c.add_argument("--tasks", required=True)
    c.add_argument("--gen", required=True)
    c.add_argument("--out", default="")
    c.add_argument("--per-test", type=float, default=6.0)
    a = ap.parse_args()
    {"select": select, "gen": gen, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
