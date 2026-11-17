"""Code/config fingerprint of one iteration's training, so a resumed run is the same experiment it started as.

    python environment_iteration/src/fingerprint.py check <iteration> [seed]
        First call: hash every file that can change this iteration's training numbers and write
        <run_dir>/fingerprint.json. Later calls (every resume): hash again and compare.
        Exit 0 = unchanged (or first call), 3 = something changed (paths printed).
    python environment_iteration/src/fingerprint.py diff <iteration> [seed]
        Same comparison, report only (never writes, exit 0). Use it before editing shared code.
    python environment_iteration/src/fingerprint.py imported <iteration>
        For an imported run (iteration.json "imported"): compare the current files with the manifest of the study
        that trained it (experiments/results_paper_budget/manifest/fingerprint.json).

What is hashed: the upstream code the training imports (skillmimic/ env, learning, utils, assets, cfg), rl_games, the
history encoder, the motion clip and the scene pose, the train config, this loop's launcher and task classes
(src/launch.py, src/iterlib.py, src/tasks/), the iteration's env config + extra_paths, and the resolved run.py flags.
"""
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iterlib as L  # noqa: E402

SHARED = [
    "skillmimic/run.py", "skillmimic/__init__.py",
    "skillmimic/env", "skillmimic/learning", "skillmimic/utils", "skillmimic/metric",
    "skillmimic/data/assets", "skillmimic/data/cfg",
    "skillmimic/data/motions/ParaHome/s6/object_transformations.pkl",   # desk / book scene pose of place_book
    "experiments/checkpoints/hist_encoder/parahome_hist60.ckpt",
    ".venv/lib/python3.8/site-packages/rl_games",
    "environment_iteration/src/launch.py", "environment_iteration/src/iterlib.py", "environment_iteration/src/tasks",
]
SKIP_DIRS = {"__pycache__", ".git"}


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(rel):
    p = os.path.join(L.REPO, rel)
    if os.path.isfile(p) or not os.path.isdir(p):
        yield rel
        return
    for root, dirs, files in os.walk(p, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            if not f.endswith((".pyc", ".tmp")):
                yield os.path.relpath(os.path.join(root, f), L.REPO)


def dependency_paths(it):
    deps = list(SHARED) + [it["cfg_env"], it["cfg_train"], it["motion_file"]] + list(it.get("extra_paths", []))
    return list(dict.fromkeys(deps))


def fingerprint(it):
    files = {}
    for rel in dependency_paths(it):
        for f in iter_files(rel):
            p = os.path.join(L.REPO, f)
            files[f] = sha256(p) if os.path.isfile(p) else "MISSING"
    files["<run.py flags>"] = hashlib.sha256(" ".join(L.train_args(it)).encode()).hexdigest()
    digest = hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()).hexdigest()
    return digest, files


def compare(ref, files):
    changed = sorted(k for k in set(ref) | set(files) if ref.get(k) != files.get(k))
    for k in changed:
        what = "added" if k not in ref else "removed" if k not in files else "changed"
        print(f"  {what:8s} {k}")
    return changed


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("check", "diff", "imported"):
        sys.exit(__doc__)
    cmd, name = sys.argv[1], sys.argv[2]
    it = L.load(name)
    if cmd == "imported":
        man = os.path.join(L.REPO, "experiments/results_paper_budget/manifest/fingerprint.json")
        ref = L.read_json(man, {}).get("files", {})
        if not it.get("imported") or not ref:
            sys.exit(f"{name} is not an imported iteration, or {L.rel(man)} is missing")
        _, files = fingerprint(it)
        orig_cfg = it["imported"].get("env_cfg_original")
        if orig_cfg and orig_cfg in ref:   # the snapshot env.yaml stands in for the config the run was trained with
            files[orig_cfg] = files.get(it["cfg_env"])
        common = {k: v for k, v in files.items() if k in ref}
        changed = compare({k: ref[k] for k in common}, common)
        print(f"[fingerprint] {name}: {len(common)} files shared with the training manifest, {len(changed)} differ")
        return 0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_dir = L.paths(name, seed)["run_dir"]
    fp_path = os.path.join(run_dir, "fingerprint.json")
    digest, files = fingerprint(it)
    old = L.read_json(fp_path)
    if old is None:
        if cmd == "diff":
            print(f"[fingerprint] {name}: no recorded fingerprint yet ({L.rel(fp_path)})")
            return 0
        L.write_json(fp_path, dict(created_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                   iteration=name, digest=digest, files=files))
        print(f"[fingerprint] {name}: recorded {digest[:12]} ({len(files)} files) -> {L.rel(fp_path)}")
        return 0
    if old["digest"] == digest:
        print(f"[fingerprint] {name}: unchanged since the first launch ({digest[:12]}, {len(files)} files)")
        return 0
    print(f"[fingerprint] {name}: files that affect training changed since the first launch:")
    compare(old["files"], files)
    return 3 if cmd == "check" else 0


if __name__ == "__main__":
    sys.exit(main())
