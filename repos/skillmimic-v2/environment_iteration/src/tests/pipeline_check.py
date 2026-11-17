"""End-to-end pipeline check of one iteration in a scratch folder (GPU, ~5-8 min). Never touches real iterations.

    python environment_iteration/src/tests/pipeline_check.py <iteration> [--keep]

Copies <iteration>'s definition (iteration.json; its env.yaml / task classes are used in place) into
environment_iteration/.check/<time>/iterations/<iteration>/, then with EI_ITER_DIR pointing there and tiny test
overrides (20 epochs, 512 training envs = the smallest that fits PPO minibatch 16384 / 32 steps,
4 eval envs, 1 eval seed):
  1. scripts/train.sh in its own process group; Ctrl-C (SIGINT to the group) once the first checkpoint exists
  2. the same command again: must resume (rl_games "resumed full training state"), reach epoch 20, then run
     eval.sh -> analyze -> status by itself
  3. the same command a third time: must skip training (already complete)
  4. scripts/video.sh ad hoc window on 2 envs: must write videos + INDEX.md
PASS/FAIL per step; the scratch folder is deleted on PASS unless --keep.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import iterlib as L  # noqa: E402


def run(cmd, env, log, timeout=1800):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n"); f.flush()
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=L.REPO, timeout=timeout).returncode


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    name, keep = sys.argv[1], "--keep" in sys.argv
    src = L.load(name)
    root = os.path.join(L.EI, ".check", time.strftime("%Y%m%d_%H%M%S"))
    it_dir = os.path.join(root, "iterations", name)
    os.makedirs(it_dir)
    it = {k: v for k, v in src.items() if k != "imported"}
    with open(os.path.join(it_dir, "iteration.json"), "w") as f:
        json.dump(it, f, indent=1)
    env = dict(os.environ, EI_ITER_DIR=os.path.join(root, "iterations"), EI_TEST_EPOCHS="20",
               EI_TEST_NUM_ENVS="512", EI_TEST_EVAL_ENVS="4", PYTHONUNBUFFERED="1")
    log = os.path.join(root, "check.log")
    train = ["bash", os.path.join(L.EI, "scripts", "train.sh"), name]
    run_dir = os.path.join(it_dir, "runs", L.run_name(name, L.protocol()["train"]["seed"]))
    ok = True

    def step(label, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[pipeline_check] {'PASS' if cond else 'FAIL'}  {label}{(' - ' + extra) if extra else ''}", flush=True)

    print(f"[pipeline_check] scratch: {L.rel(root)}  (log: {L.rel(log)})", flush=True)
    # 1. start, interrupt after the first checkpoint
    with open(log, "a") as f:
        p = subprocess.Popen(train, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=L.REPO, start_new_session=True,
                             preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    t0 = time.time()
    while time.time() - t0 < 900 and p.poll() is None and not glob.glob(os.path.join(run_dir, "nn", "*_000000[1-9]0.pth")):
        time.sleep(2)
    first = L.latest_epoch(run_dir)
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
    try:
        code = p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); code = -9
    step("interrupt", first > 0 and first < 20 and code == 130, f"checkpoint at epoch {first}, exit {code}")
    # 2. resume to the end (+ automatic eval / analysis)
    code = run(train, env, log)
    tl = open(os.path.join(run_dir, "train.log")).read() if os.path.isfile(os.path.join(run_dir, "train.log")) else ""
    step("resume", code == 0 and "resumed full training state" in tl and L.latest_epoch(run_dir) == 20,
         f"exit {code}, latest epoch {L.latest_epoch(run_dir)}")
    rep = os.path.join(it_dir, "analysis", "e20", "REPORT.md")
    step("eval + analysis", os.path.isfile(rep) and os.path.isfile(os.path.join(it_dir, "eval", "curve", "e20.json")),
         L.rel(rep))
    # 3. re-run: nothing to train
    n_seg = len(json.load(open(os.path.join(run_dir, "run_meta.json")))["segments"])
    code = run(train, env, log)
    n_seg2 = len(json.load(open(os.path.join(run_dir, "run_meta.json")))["segments"])
    step("re-run skips training", code == 0 and n_seg2 == n_seg, f"exit {code}, segments {n_seg} -> {n_seg2}")
    # 4. video
    code = run(["bash", os.path.join(L.EI, "scripts", "video.sh"), name, "--cell", "stoch", "--seed", "101",
                "--num_envs", "2", "--horizon", "40", "--window", "10:30", "--epoch", "20"], env, log)
    vids = glob.glob(os.path.join(root, "video", name, "*", "clips", "*.mp4"))
    step("video", code == 0 and len(vids) == 2, f"{len(vids)} clips")
    fp = os.path.join(run_dir, "fingerprint.json")
    step("fingerprint recorded", os.path.isfile(fp))
    print(f"[pipeline_check] {'PASS' if ok else 'FAIL'}: {name}", flush=True)
    if ok and not keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"[pipeline_check] kept {L.rel(root)} (see check.log)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
