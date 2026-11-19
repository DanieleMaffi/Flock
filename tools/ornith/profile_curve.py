"""The expert-usage curve of a strata-q36 profile (STRATA_Q36_PROFILE_OUT): per layer, the share of the router's picks the
N most-used experts take, averaged over the main layers (min and max beside it), and how many experts per layer cover a
given share.  It says how far partial offload can go: every expert a slot holds is work the CPU does not do.

    python3 tools/ornith/profile_curve.py ornith.profile
"""
import sys


def load(path):
    with open(path) as f:
        head = f.readline().split()
        if len(head) != 4 or head[0] != "strata-q36-profile" or head[1] != "1":
            raise SystemExit(f"{path}: not a strata-q36 profile")
        n_state, ne = int(head[2]), int(head[3])
        rows = [[float(v) for v in f.readline().split()] for _ in range(n_state)]
    if any(len(r) != ne for r in rows):
        raise SystemExit(f"{path}: a line does not have {ne} counts")
    return rows


def shares(row):
    """Cumulative share of the picks taken by the top 1 .. n experts."""
    tot = sum(row) or 1.0
    out, acc = [], 0.0
    for v in sorted(row, reverse=True):
        acc += v
        out.append(acc / tot)
    return out


def main():
    rows = load(sys.argv[1])
    main_layers = [r for r in rows[:-1] if sum(r) > 0] if len(rows) > 1 else rows   # (the last line: the MTP block)
    ne = len(rows[0])
    curves = [shares(r) for r in main_layers]
    picks = sum(sum(r) for r in main_layers)
    print(f"{len(main_layers)} layers, {ne} experts each, {picks:.0f} picks counted")
    print("experts in VRAM per layer -> share of the work on the GPU (mean, min, max over the layers)")
    for n in (16, 32, 48, 64, 96, 128, 160, 192, 224, ne):
        if n > ne:
            continue
        v = [c[n - 1] for c in curves]
        print(f"  {n:4d} ({100 * n / ne:5.1f}%)  {100 * sum(v) / len(v):5.1f}%   {100 * min(v):5.1f}% .. {100 * max(v):5.1f}%")
    print("experts per layer needed for a share of the work (mean over the layers)")
    for target in (0.5, 0.8, 0.9, 0.95, 0.99):
        need = [next(i + 1 for i, s in enumerate(c) if s >= target) for c in curves]
        print(f"  {100 * target:4.0f}%  {sum(need) / len(need):6.1f}   (max {max(need)})")


if __name__ == "__main__":
    main()
