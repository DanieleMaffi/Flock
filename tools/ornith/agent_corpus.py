"""Agent-like prompts for an expert-usage profile (STRATA_Q36_PROFILE_OUT): pieces of build logs, engine logs and
source code, each wrapped as a chat turn asking what an agent would ask about it, tokenized by a running llama-server
of the same model (/tokenize).  Writes one prompt per line in q36_solo_gate.py's format (comma-separated ids).

    python3 tools/ornith/agent_corpus.py --url http://127.0.0.1:8081 --out agent.ids file1.log file2.cu ...
"""
import argparse
import json
import urllib.request

ASKS = [
    "Here is the output of a command I ran. What went wrong, and what should I try next?",
    "Summarize what this log shows in a few bullet points, and point out anything unusual.",
    "Read this code and explain what it does. Is there a bug?",
    "This is tool output from my last step. Extract the errors and warnings and tell me which matters most.",
]


def tokenize(url: str, text: str) -> list:
    body = json.dumps({"content": text, "add_special": False, "parse_special": True}).encode()
    req = urllib.request.Request(url + "/tokenize", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chars", type=int, default=6000, help="characters of each file piece")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    lines = []
    for i, path in enumerate(a.files):
        text = open(path, errors="replace").read()
        for j in range(0, min(len(text), 4 * a.chars), a.chars):
            piece = text[j:j + a.chars]
            if len(piece) < 500:
                continue
            ask = ASKS[(i + j // a.chars) % len(ASKS)]
            chat = f"<|im_start|>user\n{ask}\n\n```\n{piece}\n```<|im_end|>\n<|im_start|>assistant\n"
            lines.append(",".join(str(t) for t in tokenize(a.url, chat)))
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    n = sum(len(l.split(",")) for l in lines)
    print(f"{len(lines)} prompts, {n} tokens")


if __name__ == "__main__":
    main()
