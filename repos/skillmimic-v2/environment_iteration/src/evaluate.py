"""Standard evaluation of one iteration (protocol.json "eval"; LOOP.md 3). GPU, a few minutes.

    python environment_iteration/src/evaluate.py <iteration> --part curve [--force]
    python environment_iteration/src/evaluate.py <iteration> --part final [--epoch E] [--force]

  curve  det, 8 envs, eval seed 0, at every protocol curve epoch that has a checkpoint
  final  at the budget epoch (protocol train.epochs, or --epoch): every protocol "final" cell, e.g.
         stoch 32 envs x seeds 101,102,103 and the same with the eps-NSR object perturbation (96 episodes per cell)
One Isaac Gym environment per process, so the two parts are separate calls (scripts/eval.sh runs both).
Clean frame-2 starts (protocol eval.state_init_random_prob = 0, see the note there).

Output (existing results are skipped unless --force):
  iterations/<it>/eval/curve/e<epoch>.json|.npz
  iterations/<it>/eval/<cell>/e<epoch>_s<seed>.json|.npz
  .json = meta + aggregate + episodes (evaluate.py metrics + failure events, src/metrics.py); .npz = full trace
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import iterlib as L  # noqa: E402
import metrics as MX  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=L.REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def result_path(eval_dir, cell, epoch, seed=None):
    name = f"e{epoch}" if seed is None else f"e{epoch}_s{seed}"
    return os.path.join(eval_dir, cell, name + ".json")


def score(env, S, info, extra):
    import numpy as np
    c = dict(n_clip=env.n_clip, obj_z0=env.obj_z0, ref_lift=env.ref_lift, thr=env.hold_thr,
             paper_z=env.paper_z, paper_need=env.paper_need,
             box=MX.load_box(L.REPO, env.it.get("object_box", MX.BOOK_BOX)))
    ph = MX.ref_phases(MX.clean_ref(S, c["n_clip"]), c["obj_z0"])
    eps = []
    for e in range(env.num_envs):
        m = MX.episode_metrics(S, S["fall_t"], c, e)
        f = MX.failure_events(S, S["fall_t"], c, e, ph)
        m.update({k: f[k] for k in ("mode", "detail", "first_bad", "phase", "grasp_frame", "drop_frame",
                                    "off_desk_frame", "rot_off_frame", "floor_frame", "hold_streaks", "events")})
        m["env"] = e
        m["start_ref_frame"] = int(S["ref_frame"][0, e])
        eps.append(m)
    agg = MX.aggregate(eps)
    agg["modes"] = {k: sum(e["mode"] == k for e in eps) for k in MX.MODES}
    meta = dict(env.meta(), **info, **extra, ref_phases=ph, protocol_version=L.protocol()["version"],
                git=git_commit(), evaluated_at=datetime.datetime.now().isoformat(timespec="seconds"))
    return dict(meta=meta, aggregate=agg, episodes=eps), np


def save(path, res, S, np):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path[:-5] + ".npz", **S)          # npz first; the .json (the "done" marker) last, atomic
    L.write_json(path, res)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration")
    p.add_argument("--part", required=True, choices=["curve", "final"])
    p.add_argument("--epoch", type=int, default=None, help="final: evaluate this epoch instead of the budget")
    p.add_argument("--seed_run", type=int, default=None, help="training seed (default protocol train.seed)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    pr = L.protocol()
    ev = pr["eval"]
    P = L.paths(a.iteration, a.seed_run)
    if not L.checkpoints(P["run_dir"]):
        sys.exit(f"[eval] no checkpoints in {L.rel(P['run_dir'])}/nn")

    if a.part == "curve":
        cv = ev["curve"]
        jobs = []
        for ep in cv["epochs"]:
            ck = L.checkpoint_at(P["run_dir"], ep)
            out = result_path(P["eval_dir"], "curve", ep)
            if ck is None:
                print(f"[eval] curve epoch {ep}: no checkpoint, skipped")
            elif a.force or not os.path.isfile(out):
                jobs.append((ep, ck, out, cv["seed"], False, "curve"))
        mode, n_envs = cv["mode"], cv["num_envs"]
    else:
        ep = a.epoch or pr["train"]["epochs"]
        ck = L.checkpoint_at(P["run_dir"], ep)
        if ck is None:
            sys.exit(f"[eval] final: no checkpoint for epoch {ep} in {L.rel(P['run_dir'])}/nn "
                     f"(latest {L.latest_epoch(P['run_dir'])})")
        jobs, modes = [], set()
        for cell in ev["final"]:
            modes.add((cell["mode"], cell["num_envs"]))
            for s in cell["seeds"]:
                out = result_path(P["eval_dir"], cell["cell"], ep, s)
                if a.force or not os.path.isfile(out):
                    jobs.append((ep, ck, out, s, bool(cell["perturb"]), cell["cell"]))
        if len(modes) != 1:
            sys.exit("[eval] all final cells must share mode and num_envs (one environment per process)")
        mode, n_envs = modes.pop()
    if not jobs:
        print(f"[eval] {a.iteration} {a.part}: all done")
        return
    import rollout as RO
    env = RO.Env(a.iteration, n_envs, mode, ev["horizon"], ev["start_frame"], seed=jobs[0][3],
                 first_checkpoint=jobs[0][1])
    print(f"[eval] {a.iteration} {a.part}: {len(jobs)} rollout(s), {n_envs} envs, {mode}")
    for k, (ep, ck, out, seed, perturb, cell) in enumerate(jobs):
        t0 = time.time()
        S, info = env.rollout(ck, seed, perturb)
        res, np = score(env, S, info, dict(cell=cell, seed=seed, perturb=perturb))
        save(out, res, S, np)
        g = res["aggregate"]
        top = ", ".join(f"{m} {n}" for m, n in sorted(g["modes"].items(), key=lambda x: -x[1]) if n)
        print(f"[eval {k + 1}/{len(jobs)}] {cell} e{ep} s{seed}: task={g['task_success']:.2f} "
              f"paper={g['paper_success']:.2f} lift={g['lift_success']:.2f} progress={g['task_progress']:.2f} "
              f"| {top} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
