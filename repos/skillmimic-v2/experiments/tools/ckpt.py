"""Pick the checkpoint a run resumes from (used by experiments/scripts/train.sh).

    python experiments/tools/ckpt.py latest <run_dir>     # prints "<path>\t<epoch>", nothing if there is none

Every .pth in <run_dir>/nn is a full training state (weights, normalizers, optimizer, epoch, frame, ATS state, RNG):
numbered `<run>_00000250.pth`, best-reward `<run>_e123_r7.9147.pth`, `<run>_e2500.pth`, and the plain latest
`<run>.pth`. The one with the highest epoch that loads cleanly wins. Saves are atomic (write .tmp, then rename),
so a file that fails to load can only come from an older code version or a broken disk; it is skipped with a
warning and the next newest is used. Leftover .pth.tmp files from an interrupted save are deleted.
"""
import glob
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # torch.load weights_only notice

NUM_RE = re.compile(r"_(\d{8})\.pth$")
BEST_RE = re.compile(r"_e(\d+)(?:_r-?[\d.]+)?\.pth$")


def name_epoch(path):
    m = NUM_RE.search(path) or BEST_RE.search(path)
    return int(m.group(1)) if m else None


def load_ok(path):
    import torch
    try:
        st = torch.load(path, map_location="cpu")
        return int(st["epoch"]) if all(k in st for k in ("model", "optimizer", "epoch")) else None
    except Exception as e:  # truncated / corrupt file
        print(f"[ckpt] cannot load {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def latest(run_dir):
    nn = os.path.join(run_dir, "nn")
    for tmp in glob.glob(os.path.join(nn, "*.pth.tmp")):
        print(f"[ckpt] removing unfinished save {tmp}", file=sys.stderr)
        os.remove(tmp)
    plain = os.path.join(nn, os.path.basename(os.path.normpath(run_dir)) + ".pth")
    cands = []
    for p in glob.glob(os.path.join(nn, "*.pth")):
        if p == plain:
            continue
        e = name_epoch(p)
        if e is not None:
            cands.append((e, 1 if NUM_RE.search(p) else 0, p))
    if os.path.isfile(plain):
        e = load_ok(plain)
        if e is not None:
            cands.append((e, 2, plain))  # already verified; preferred on ties
    for e, prio, p in sorted(cands, reverse=True):
        got = e if prio == 2 else load_ok(p)
        if got is not None:
            return p, got
    return None, None


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "latest":
        sys.exit(__doc__)
    path, epoch = latest(sys.argv[2])
    if path:
        print(f"{path}\t{epoch}")
