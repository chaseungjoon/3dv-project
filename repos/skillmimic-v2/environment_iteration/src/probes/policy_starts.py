"""Where along the clip can a trained policy keep the object? (GPU, ~30 s per start frame)

    python environment_iteration/src/probes/policy_starts.py <iteration> --start 50,70,85 --epochs 2000,3000

Starts every env at reference frame `start` with the reference state (body AND object as in the demonstration,
like a training RIS start, but clean), runs `--horizon` steps (default 60 = the training episode length) with the
iteration's own policy, and prints per checkpoint: frames until the object is > 0.25 m from its reference (the
training early termination, env.yaml objTerminationDist) and > 0.10 m (the task criterion), falls, and the return
until the training episode would end. Separates "cannot pick it up" (early starts fail) from "cannot carry / walk"
(late starts fail). One environment per process: each start frame runs in its own subprocess.
Used for fix-2 ITERATION.md section 8 (probe 1).
"""
import argparse
import os
import subprocess
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)


def one(it, start, epochs, n, horizon, seed, term):
    import numpy as np
    import iterlib as L
    import rollout as RO
    ck = dict(L.checkpoints(L.paths(it)["run_dir"]))
    env = RO.Env(it, n, "stoch", horizon, start_frame=start, seed=seed, first_checkpoint=ck[epochs[0]])
    for ep in epochs:
        S, _ = env.rollout(ck[ep], seed)
        err = np.linalg.norm(S["obj"] - S["ref_obj"], axis=-1)
        T, E = err.shape

        def first(th):
            return np.array([np.argmax(err[:, e] > th) if (err[:, e] > th).any() else T for e in range(E)])
        t_term, t10 = first(term), first(0.10)
        fall = np.where(S["fall_t"] < 0, T, S["fall_t"])
        end = np.minimum(t_term, fall)
        ret = np.array([S["reward"][:end[e], e].sum() for e in range(E)])
        print(f"[starts] {it} e{ep} start {start:3d}: object > {term:.2f} m after {np.median(t_term):3.0f} frames "
              f"(p25 {np.percentile(t_term, 25):.0f}, p75 {np.percentile(t_term, 75):.0f}) | > 0.10 m after "
              f"{np.median(t10):3.0f} | fell {np.mean(S['fall_t'] >= 0) * 100:3.0f}% | return {ret.mean():5.1f}",
              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration")
    p.add_argument("--start", default="50,55,60,65,70,85,100,115", help="reference frames")
    p.add_argument("--epochs", default="3000")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--term", type=float, default=0.25, help="object distance of the training termination (m)")
    p.add_argument("--_one", type=int, default=-1, help=argparse.SUPPRESS)
    a = p.parse_args()
    epochs = [int(x) for x in a.epochs.split(",")]
    if a._one >= 0:
        return one(a.iteration, a._one, epochs, a.num_envs, a.horizon, a.seed, a.term)
    for s in [int(x) for x in a.start.split(",")]:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), a.iteration, "--epochs", a.epochs,
                            "--num_envs", str(a.num_envs), "--horizon", str(a.horizon), "--seed", str(a.seed),
                            "--term", str(a.term), "--_one", str(s)], capture_output=True, text=True)
        out = [ln for ln in r.stdout.splitlines() if ln.startswith("[starts]")]
        print("\n".join(out) if out else f"[starts] start {s}: failed\n{r.stderr[-1500:]}", flush=True)


if __name__ == "__main__":
    main()
