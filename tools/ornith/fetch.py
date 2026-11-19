"""Downloads one size of Unsloth's Ornith-1.5-122B (and the picture reader) at a pinned commit into a folder, resuming
partial files and checking each one's size and SHA-256.  Used by flock's compose (docs/ORNITH.md).

    python3 tools/ornith/fetch.py /data/Strata-data/models/ornith-ud-q4_k_xl              # UD-Q4_K_XL (80 GB)
    python3 tools/ornith/fetch.py /data/Strata-data/models/ornith-ud-iq2_xxs UD-IQ2_XXS   # 42 GB
"""
import hashlib
import os
import signal
import sys
import urllib.request

if hasattr(signal, "SIGPIPE"):   # a closed log (the container's output) must not stop a download: progress lines are best effort
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def say(line: str) -> None:
    try:
        print(line, flush=True)
    except OSError:
        pass

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
    "UD-Q3_K_XL": {
        "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf": (10942929, "d8a23c487741fdac7026e10013fa8e320502cf01048a2333b2db8ba2a9f74fe5"),
        "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00002-of-00003.gguf": (49791670080, "95f76e67ffc49e2b0b4a4dfe5c42269b4b275ecf9e3696f599f3eeac92180323"),
        "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00003-of-00003.gguf": (9958641536, "1c4b52bbb1d45a21b5b7927429a5c80f264c688f2d14ed50ba0b6da0260d4028"),
        **MMPROJ},
    "Q8_0": {   # practically lossless (Unsloth: KL divergence about 0.018): the quality reference and a source to quantize from
        "Q8_0/Ornith-1.5-122B-Q8_0-00001-of-00004.gguf": (10942929, "e9dbc5401757506c6bbb04dba3bf65decc6435aaf0733f1e95cc58a9332166fc"),
        "Q8_0/Ornith-1.5-122B-Q8_0-00002-of-00004.gguf": (49855668288, "c1ea340f63c932895417f7c2e7c4967b4b56f3e5a3f981242d0482c662aae2b8"),
        "Q8_0/Ornith-1.5-122B-Q8_0-00003-of-00004.gguf": (49903347168, "644c26a3d8cab405186ef545e4071b433a23c2a359e45434c822d9c48884eb4c"),
        "Q8_0/Ornith-1.5-122B-Q8_0-00004-of-00004.gguf": (32785752352, "43bdc1b0cb764b5531d320816fb6004245256a414bfd55f1fdb69e3db42eaa57"),
        **MMPROJ},
}


# a mistyped pin would delete a good file: every one must be a whole SHA-256 (64 hex digits)
for _files in SIZES.values():
    for _name, (_size, _sha) in _files.items():
        assert len(_sha) == 64 and all(c in "0123456789abcdef" for c in _sha), f"bad SHA-256 pin for {_name}"


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
                    say(f"{os.path.basename(out)}: {done / 1e9:.1f} / {size / 1e9:.1f} GB")
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
    say(f"ok {out}")


if __name__ == "__main__":
    dest = sys.argv[1]
    size_name = sys.argv[2] if len(sys.argv) > 2 else "UD-Q4_K_XL"
    if size_name not in SIZES:
        raise SystemExit(f"no size {size_name}: one of {', '.join(SIZES)}")
    os.makedirs(dest, exist_ok=True)
    for remote, (size, sha) in SIZES[size_name].items():
        fetch(remote, os.path.join(dest, os.path.basename(remote)), size, sha)
    say("all files ok")
