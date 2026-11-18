"""Run a trained policy zero-shot in a modified environment. (GPU, ~1 min)

    python environment_iteration/src/probes/policy_env.py fix-1 --epochs 3000 --env fix-2 \
        --set 'sim.physx.contact_offset=0.02' --seeds 101,102,103

Builds the environment of iteration `--env` (default: the policy's own; its env.yaml and task class), applies `--set` edits to a scratch copy of
its env.yaml, and rolls out the policy of `iteration` from the protocol's clean frame-2 start (stoch, `--num_envs`
per seed). Prints lift success, held frames, drop events, falls in the clip and failure modes (src/metrics.py).
Observation and action sizes must match. Answers "does this physics change keep / break a grasp that a policy
already found", which a perfect-imitator probe cannot (the demonstrated pickup is not executable open-loop).
Used for fix-2 ITERATION.md section 8 (probe 3); one environment per process, so one `--set` per call.
"""
import argparse
import collections
import os
import sys
import tempfile

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration", help="whose policy")
    p.add_argument("--epochs", default="3000")
    p.add_argument("--env", default="", help="whose environment (default: the policy's)")
    p.add_argument("--set", default="", help="env.yaml edits, e.g. 'sim.physx.contact_offset=0.02;sim.substeps=4'")
    p.add_argument("--seeds", default="101")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--horizon", type=int, default=160)
    a = p.parse_args()
    import numpy as np
    import yaml
    import iterlib as L
    import metrics as MX
    import rollout as RO
    env_it = a.env or a.iteration
    cfg = yaml.safe_load(open(os.path.join(L.REPO, L.load(env_it)["cfg_env"])))
    for kv in [x for x in a.set.split(";") if x.strip()]:
        k, v = kv.split("=")
        node = cfg
        ks = k.strip().split(".")
        for kk in ks[:-1]:
            node = node[kk]
        node[ks[-1]] = yaml.safe_load(v)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.environ.get("TMPDIR", "/tmp"))
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    run_dir = L.paths(a.iteration)["run_dir"]
    epochs = [int(x) for x in a.epochs.split(",")]
    cks = {e: L.checkpoint_at(run_dir, e) for e in epochs}
    seeds = [int(x) for x in a.seeds.split(",")]
    env = RO.Env(a.iteration, a.num_envs, "stoch", a.horizon, seed=seeds[0], first_checkpoint=cks[epochs[0]],
                 overrides={"cfg_env": tmp.name, "task": L.load(env_it)["task"]})
    c = RO.consts(env)
    for ep in epochs:
        rows = []
        for sd in seeds:
            S, _ = env.rollout(cks[ep], sd)
            m = [MX.episode_metrics(S, S["fall_t"], c, i) for i in range(a.num_envs)]
            ph = MX.ref_phases(MX.clean_ref(S, c["n_clip"]), c["obj_z0"])
            fe = [MX.failure_events(S, S["fall_t"], c, i, ph) for i in range(a.num_envs)]
            g = MX.aggregate(m)
            drop = np.mean([any(k == "drop" for _, k, _ in f["events"]) for f in fe])
            modes = collections.Counter(f["mode"] for f in fe)
            rows.append((g["lift_success"], g["held_frames"], drop, g["fell_in_clip"]))
            print(f"[policy_env] {a.iteration} e{ep} in {env_it} [{a.set}] seed {sd}: lift {g['lift_success'] * 100:3.0f}% "
                  f"held {g['held_frames']:5.1f} drop {drop * 100:3.0f}% fell-in-clip {g['fell_in_clip'] * 100:3.0f}% "
                  f"progress {g['task_progress'] * 100:3.0f}% | {dict(modes)}", flush=True)
        r = np.mean(rows, 0)
        print(f"[policy_env] {a.iteration} e{ep} in {env_it} [{a.set}] mean of {len(seeds)} seed(s) x {a.num_envs}: "
              f"lift {r[0] * 100:.0f}% held {r[1]:.1f} drop {r[2] * 100:.0f}% fell-in-clip {r[3] * 100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
