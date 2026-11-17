"""Shared helpers for the environment-iteration loop (environment_iteration/LOOP.md).

Everything that needs to know "what is iteration fix-N and where are its files" goes through this module, so the
paths and the run.py arguments are defined in exactly one place.

    python environment_iteration/src/iterlib.py list                 # iterations, newest last
    python environment_iteration/src/iterlib.py args fix-2 [train]   # run.py flags of an iteration
    python environment_iteration/src/iterlib.py paths fix-2          # its folders (shell-friendly KEY=VALUE)
    python environment_iteration/src/iterlib.py latest               # newest iteration name
    python environment_iteration/src/iterlib.py proto                # protocol values (KEY=VALUE)

Stdlib only (no torch / isaacgym), so the shell scripts can call it cheaply.
"""
import glob
import json
import os
import re
import sys

EI = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # environment_iteration/
REPO = os.path.dirname(EI)                                            # repos/skillmimic-v2/
REAL_ITER_DIR = os.path.join(EI, "iterations")
# Test mode (scripts/check.sh --gpu): EI_ITER_DIR points at a scratch folder, and only then the EI_TEST_* overrides
# below are honoured, so a pipeline test can never write into, or shrink the protocol of, the real iterations.
ITER_DIR = os.path.abspath(os.environ.get("EI_ITER_DIR") or REAL_ITER_DIR)
TEST_MODE = ITER_DIR != REAL_ITER_DIR
VIDEO_DIR = os.path.join(os.path.dirname(ITER_DIR), "video") if TEST_MODE else os.path.join(EI, "video")
NAME_RE = re.compile(r"^fix-(\d+)$")
CKPT_RE = re.compile(r"_(\d{8})\.pth$")


def protocol():
    with open(os.path.join(EI, "protocol.json")) as f:
        p = json.load(f)
    if TEST_MODE:   # EI_TEST_EPOCHS / EI_TEST_NUM_ENVS / EI_TEST_EVAL_ENVS: tiny pipeline test only
        ep = os.environ.get("EI_TEST_EPOCHS")
        if ep:
            p["train"]["epochs"] = int(ep)
            p["eval"]["curve"]["epochs"] = sorted({int(ep) // 2 // 10 * 10, int(ep)} - {0})
        if os.environ.get("EI_TEST_NUM_ENVS"):
            p["train"]["num_envs"] = int(os.environ["EI_TEST_NUM_ENVS"])
        if os.environ.get("EI_TEST_EVAL_ENVS"):
            n = int(os.environ["EI_TEST_EVAL_ENVS"])
            p["eval"]["curve"]["num_envs"] = n
            for c in p["eval"]["final"]:
                c["num_envs"], c["seeds"] = n, c["seeds"][:1]
    elif any(k.startswith("EI_TEST_") for k in os.environ):
        raise SystemExit("EI_TEST_* overrides need EI_ITER_DIR outside environment_iteration/iterations "
                         "(refusing to change the protocol of the real iterations)")
    return p


def iter_number(name):
    m = NAME_RE.match(name)
    if not m:
        raise SystemExit(f"bad iteration name {name!r}: must be fix-<n> (e.g. fix-2)")
    return int(m.group(1))


def list_iterations():
    names = [os.path.basename(p) for p in glob.glob(os.path.join(ITER_DIR, "fix-*"))
             if os.path.isfile(os.path.join(p, "iteration.json")) and NAME_RE.match(os.path.basename(p))]
    return sorted(names, key=iter_number)


def load(name):
    iter_number(name)
    p = os.path.join(ITER_DIR, name, "iteration.json")
    if not os.path.isfile(p):
        raise SystemExit(f"no iteration {name}: {os.path.relpath(p, REPO)} missing "
                         f"(create it with environment_iteration/scripts/new_iteration.sh)")
    with open(p) as f:
        it = json.load(f)
    if it.get("name") != name:
        raise SystemExit(f"{p}: name field {it.get('name')!r} != folder {name!r}")
    return it


def run_name(name, seed):
    return f"{name}_s{seed}"


def paths(name, seed=None):
    """All folders of one iteration. run_dir is the folder that holds nn/ (checkpoints) for that seed."""
    seed = protocol()["train"]["seed"] if seed is None else seed
    d = os.path.join(ITER_DIR, name)
    rn = run_name(name, seed)
    return dict(
        iter_dir=d,
        runs_dir=os.path.join(d, "runs"),
        run_name=rn,
        run_dir=os.path.join(d, "runs", rn),
        eval_dir=os.path.join(d, "eval"),
        analysis_dir=os.path.join(d, "analysis"),
        logs_dir=os.path.join(d, "logs"),
        video_dir=os.path.join(VIDEO_DIR, name),
        clips=os.path.join(d, "clips.json"),
        notes=os.path.join(d, "ITERATION.md"),
    )


def base_args(it):
    """run.py flags shared by training and evaluation (same order as experiments/tools/methods.py base_args)."""
    for k in ("task", "cfg_env", "cfg_train", "asset", "motion_file"):
        if not it.get(k):
            raise SystemExit(f"{it['name']}: iteration.json needs '{k}'")
        if k in ("cfg_env", "cfg_train", "motion_file") and not os.path.exists(os.path.join(REPO, it[k])):
            raise SystemExit(f"{it['name']}: {k} {it[k]} does not exist")
    return ["--task", it["task"], "--cfg_env", it["cfg_env"], "--cfg_train", it["cfg_train"],
            "--asset_file_name", it["asset"], "--motion_file", it["motion_file"]] + list(it.get("common_args", []))


def train_args(it):
    return base_args(it) + list(it.get("train_args", []))


def checkpoints(run_dir):
    """Numbered checkpoints [(epoch, path)] sorted by epoch."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "nn", "*.pth")):
        m = CKPT_RE.search(p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def checkpoint_at(run_dir, epoch):
    for e, p in checkpoints(run_dir):
        if e == epoch:
            return p
    return None


def latest_epoch(run_dir):
    c = checkpoints(run_dir)
    return c[-1][0] if c else 0


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    """Atomic write (tmp + rename): a reader never sees a half-written file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, path)


def rel(p):
    return os.path.relpath(p, REPO)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["list"]:
        for n in list_iterations():
            print(n, "\t", load(n).get("summary", ""))
    elif a[:1] == ["latest"]:
        names = list_iterations()
        print(names[-1] if names else "")
    elif a[:1] == ["args"] and len(a) in (2, 3):
        it = load(a[1])
        print(" ".join(train_args(it) if a[2:] == ["train"] else base_args(it)))
    elif a[:1] == ["proto"]:   # shell-friendly protocol values (scripts/train.sh)
        p = protocol()
        t = p["train"]
        keep = sorted(set(p["eval"]["curve"]["epochs"]) | {t["epochs"]})
        print(f"EPOCHS={t['epochs']}\nNUM_ENVS={t['num_envs']}\nSEED={t['seed']}\nRETRIES={t['retries']}\n"
              f"KEEP_NEWEST={t['keep_newest']}\nPRUNE_EVERY={t['prune_every_s']}\n"
              f"KEEP_EPOCHS={','.join(map(str, keep))}\nTEST_MODE={int(TEST_MODE)}")
    elif a[:1] == ["paths"] and len(a) in (2, 3):
        load(a[1])
        for k, v in paths(a[1], int(a[2]) if len(a) == 3 else None).items():
            print(f"{k.upper()}={v}")
    else:
        sys.exit(__doc__)
