"""Downloads one size of Unsloth's Ornith-1.5-122B (and the picture reader) at a pinned commit into a folder, resuming
partial files and checking each one's size and SHA-256.  Used by flock's compose (docs/ORNITH.md).

    python3 tools/ornith/fetch.py /data/Strata-data/models/ornith-ud-q4_k_xl              # UD-Q4_K_XL (80 GB)
    python3 tools/ornith/fetch.py /data/Strata-data/models/ornith-ud-iq2_xxs UD-IQ2_XXS   # 42 GB
"""
import hashlib
import os
import sys
import urllib.request

REPO = "unsloth/Ornith-1.5-122B-GGUF"
REV = "745bd742473a6cdddb1c6f78fb9a3b3a9293bd41"
Q = "UD-Q4_K_XL/Ornith-1.5-122B-UD-Q4_K_XL"
MMPROJ = {"mmproj-F16.gguf": (908724544, "39db176176a904b101f5173a4f03e785990f9279f94ab373c4fd629ae7011817")}
SIZES = {   # size -> remote path -> (bytes, sha256), from the Hugging Face API at REV
    "UD-Q4_K_XL": {
        f"{Q}-00001-of-00003.gguf": (10942929, "7159d31a6a6ce5528500b490880383844c3c9b21fcdaad94e66c9e65233ff359"),
        f"{Q}-00002-of-00003.gguf": (49971099680, "e66952128394bb4c75554c7e10a12a8b63dd1c7f049d876a02885a946e30c1c3"),
        f"{Q}-00003-of-00003.gguf": (29731729056, "21325ec18a9e3210eeffada0fc327a25a79171918f1d04f8f0a06b61866ac9ba"),
        **MMPROJ},
    "UD-IQ2_XXS": {
        "Ornith-1.5-122B-UD-IQ2_XXS.gguf": (41967395072, "0497778d4f7f5008dec2d4b102183a699faf8433d0102dd143e86f9db13a3adc"),
        **MMPROJ},
}


def fetch(remote: str, out: str, size: int, sha: str) -> None:
    if os.path.exists(out + ".ok"):
        return
    have = os.path.getsize(out) if os.path.exists(out) else 0
    if have < size:
        url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{REPO}/resolve/{REV}/{remote}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "flock-fetch"})
        with urllib.request.urlopen(req, timeout=60) as r, open(out, "ab") as f:
            if have and r.status != 206:
                raise SystemExit(f"{remote}: the server ignored the resume request")
            done = have
            while chunk := r.read(8 << 20):
                f.write(chunk)
                done += len(chunk)
                if done % (2 << 30) < (8 << 20):
                    print(f"{os.path.basename(out)}: {done / 1e9:.1f} / {size / 1e9:.1f} GB", flush=True)
    if os.path.getsize(out) != size:
        raise SystemExit(f"{out}: {os.path.getsize(out)} bytes, expected {size}")
    h = hashlib.sha256()
    with open(out, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    if h.hexdigest() != sha:
        os.remove(out)
        raise SystemExit(f"{out}: SHA-256 mismatch, deleted; run again")
    open(out + ".ok", "w").close()
    print(f"ok {out}", flush=True)


if __name__ == "__main__":
    dest = sys.argv[1]
    size_name = sys.argv[2] if len(sys.argv) > 2 else "UD-Q4_K_XL"
    if size_name not in SIZES:
        raise SystemExit(f"no size {size_name}: one of {', '.join(SIZES)}")
    os.makedirs(dest, exist_ok=True)
    for remote, (size, sha) in SIZES[size_name].items():
        fetch(remote, os.path.join(dest, os.path.basename(remote)), size, sha)
    print("all files ok", flush=True)
