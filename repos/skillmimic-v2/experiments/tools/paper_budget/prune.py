"""Delete numbered checkpoints that the PAPER vs FIX experiment will never use (run_full.sh runs this every few minutes
while a run trains, and once more after it finishes). Stdlib only, no GPU, never touches a file it should keep.

    python experiments/tools/paper_budget/prune.py <run_dir> <keep_epochs,comma,separated> [--newest 3]

Kept in <run_dir>/nn/:
  * numbered checkpoints <run>_<epoch:08d>.pth whose epoch is in the keep list (pre-registered eval checkpoints + final)
  * the newest --newest numbered checkpoints (resume source, plus spares in case the newest one is damaged)
  * everything that is not a numbered checkpoint: <run>.pth (latest), best-reward <run>_e<N>_r<R>.pth, .tmp files
Saves are atomic (written as .tmp, then renamed), so a numbered .pth is always complete and nothing being written
is ever matched. Every deletion is appended to <run_dir>/pruned.log (epoch, bytes, time).
"""
import argparse
import datetime
import glob
import os
import re

NUM_RE = re.compile(r"_(\d{8})\.pth$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("keep")
    ap.add_argument("--newest", type=int, default=3)
    a = ap.parse_args()
    keep = {int(e) for e in a.keep.split(",") if e.strip()}
    nn = os.path.join(a.run_dir, "nn")
    numbered = sorted((int(m.group(1)), p) for p in glob.glob(os.path.join(nn, "*.pth")) if (m := NUM_RE.search(p)))
    if len(numbered) <= a.newest:
        return
    newest = {e for e, _ in numbered[-a.newest:]}
    gone, freed = [], 0
    for e, p in numbered:
        if e in keep or e in newest:
            continue
        try:
            size = os.path.getsize(p)
            os.remove(p)
        except FileNotFoundError:
            continue
        gone.append((e, size))
        freed += size
    if gone:
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with open(os.path.join(a.run_dir, "pruned.log"), "a") as f:
            for e, size in gone:
                f.write(f"{now}\tdeleted epoch {e}\t{size} bytes\n")
        print(f"[prune] {os.path.basename(os.path.normpath(a.run_dir))}: deleted {len(gone)} checkpoint(s), "
              f"freed {freed / 2**20:.0f} MiB (kept {len(keep)} eval epochs + newest {a.newest})")


if __name__ == "__main__":
    main()
