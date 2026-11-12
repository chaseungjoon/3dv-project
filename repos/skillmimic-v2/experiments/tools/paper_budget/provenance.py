"""Provenance and checkpoint checks for the PAPER vs FIX full-budget experiment (PAPER_FIX_FULL_BUDGET.md 3, 7, 13).

    python experiments/tools/paper_budget/provenance.py check <summary_dir>
        First call: hash every file that can change training/eval numbers and write <summary_dir>/manifest/fingerprint.json
        (plus git commit, git status, python/torch versions, GPU). Later calls: hash again and compare.
        Every call appends one line to <summary_dir>/manifest/invocations.jsonl.
        Exit 0 = same files (or first call), 3 = something changed (the changed paths are printed).

    python experiments/tools/paper_budget/provenance.py ckpts <summary_dir> <epochs> <run_dir> [<run_dir> ...]
        For each run: the numbered checkpoint <run>_<epochs:08d>.pth must exist and its stored epoch/frame must be
        epochs and epochs*NUM_ENVS*32. Writes <summary_dir>/checkpoints.{csv,json} with SHA-256. Exit 0 = all good.

stdlib only for `check` (torch is imported lazily by `ckpts`).
"""
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Everything a training or evaluation run reads. The orchestration scripts themselves (paper_budget/, run_full.sh)
# are not in the list: changing them cannot change a number, so it must not block a resume.
FINGERPRINT_PATHS = [
    "skillmimic/run.py", "skillmimic/__init__.py",
    "skillmimic/env", "skillmimic/learning", "skillmimic/utils", "skillmimic/metric",
    "skillmimic/data/assets", "skillmimic/data/cfg",
    "skillmimic/data/motions/ParaHome/place_book",
    "skillmimic/data/motions/ParaHome/s6/object_transformations.pkl",   # desk pose for place_book
    "experiments/configs",
    "experiments/tools/methods.py", "experiments/tools/ckpt.py", "experiments/tools/evaluate.py",
    "experiments/scripts/_common.sh", "experiments/scripts/train.sh",
    "experiments/checkpoints/hist_encoder/parahome_hist60.ckpt",
    ".venv/lib/python3.8/site-packages/rl_games",
]
SKIP_DIRS = {"__pycache__", ".git"}


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(rel):
    p = os.path.join(REPO, rel)
    if os.path.isfile(p):
        yield rel
        return
    if not os.path.isdir(p):
        yield rel  # recorded as missing
        return
    for root, dirs, files in os.walk(p, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            if f.endswith((".pyc", ".pyo")):
                continue
            yield os.path.relpath(os.path.join(root, f), REPO)


def fingerprint():
    files = {}
    for rel in FINGERPRINT_PATHS:
        for f in iter_files(rel):
            full = os.path.join(REPO, f)
            files[f] = sha256(full) if os.path.isfile(full) else "MISSING"
    digest = hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()).hexdigest()
    return digest, files


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def env_info():
    py = os.environ.get("PY", sys.executable)
    return dict(
        git_commit=sh("git rev-parse HEAD"), git_status=sh("git status --short").splitlines(),
        host=platform.node(), python=py,
        torch=sh(f"'{py}' -c 'import torch; print(torch.__version__, torch.version.cuda)'"),
        gpu=sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    )


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def cmd_check(summary_dir):
    mdir = os.path.join(summary_dir, "manifest")
    os.makedirs(mdir, exist_ok=True)
    fp_path = os.path.join(mdir, "fingerprint.json")
    digest, files = fingerprint()
    info = env_info()
    rec = dict(time=now(), digest=digest, **{k: info[k] for k in ("git_commit", "git_status")})
    code = 0
    if not os.path.isfile(fp_path):
        with open(fp_path, "w") as f:
            json.dump(dict(created_at=rec["time"], digest=digest, **info, files=files), f, indent=1)
        rec["result"] = "recorded"
        print(f"[provenance] recorded fingerprint {digest[:12]} ({len(files)} files) -> {fp_path}")
    else:
        with open(fp_path) as f:
            ref = json.load(f)
        if ref["digest"] == digest:
            rec["result"] = "match"
            print(f"[provenance] fingerprint matches the first launch ({digest[:12]}, {len(files)} files)")
        else:
            old = ref["files"]
            changed = sorted(k for k in set(old) | set(files) if old.get(k) != files.get(k))
            rec.update(result="CHANGED", changed=changed)
            print(f"[provenance] FILES CHANGED since the first launch ({ref['created_at']}):", file=sys.stderr)
            for k in changed[:50]:
                print(f"    {k}: {old.get(k, 'absent')[:12]} -> {files.get(k, 'absent')[:12]}", file=sys.stderr)
            if len(changed) > 50:
                print(f"    ... {len(changed) - 50} more", file=sys.stderr)
            code = 3
    with open(os.path.join(mdir, "invocations.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return code


def cmd_ckpts(summary_dir, epochs, run_dirs):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    rows, ok = [], True
    for rd in run_dirs:
        name = os.path.basename(os.path.normpath(rd))
        p = os.path.join(rd, "nn", f"{name}_{epochs:08d}.pth")
        row = dict(run=name, checkpoint=os.path.relpath(p, REPO) if os.path.isabs(p) else p)
        meta_p = os.path.join(rd, "run_meta.json")
        spe = 2048 * 32
        if os.path.isfile(meta_p):
            with open(meta_p) as f:
                spe = json.load(f).get("samples_per_epoch", spe)
        row["expected_frame"] = epochs * spe
        if not os.path.isfile(p):
            row.update(status="MISSING"); ok = False
        else:
            st = torch.load(p, map_location="cpu")
            row.update(epoch=int(st["epoch"]), frame=int(st.get("frame", -1)), bytes=os.path.getsize(p), sha256=sha256(p))
            del st
            good = row["epoch"] == epochs and row["frame"] == row["expected_frame"]
            row["status"] = "OK" if good else "MISMATCH"
            ok &= good
        rows.append(row)
        print(f"[ckpt] {name}: {row['status']} epoch={row.get('epoch')} frame={row.get('frame')} "
              f"(expected {epochs} / {row['expected_frame']}) sha256={row.get('sha256', '')[:16]}")
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "checkpoints.json"), "w") as f:
        json.dump(dict(checked_at=now(), epochs=epochs, runs=rows), f, indent=1)
    keys = ["run", "status", "epoch", "frame", "expected_frame", "bytes", "sha256", "checkpoint"]
    with open(os.path.join(summary_dir, "checkpoints.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        sys.exit(cmd_check(sys.argv[2]))
    if len(sys.argv) >= 5 and sys.argv[1] == "ckpts":
        sys.exit(cmd_ckpts(sys.argv[2], int(sys.argv[3]), sys.argv[4:]))
    sys.exit(__doc__)
