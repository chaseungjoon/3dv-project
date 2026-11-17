"""Start skillmimic/run.py for one iteration, with that iteration's task classes registered.

    python environment_iteration/src/launch.py train <iteration> --seed 0 --epochs 3000 --num_envs 2048 \
        [--resume_from <ckpt>]

Called by environment_iteration/scripts/train.sh (which handles resume, retries, locking and logging); do not call
it directly for a real run. src/rollout.py uses prepare() + run_main() the same way for evaluation and video.
"""
import argparse
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import iterlib as L  # noqa: E402


def prepare(it):
    """cwd = repo, skillmimic/ importable, isaacgym before torch, iteration task class registered."""
    os.chdir(L.REPO)
    sm = os.path.join(L.REPO, "skillmimic")
    if sm not in sys.path:
        sys.path.insert(0, sm)
    from isaacgym import gymapi  # noqa: F401  (must come before torch)
    import tasks
    return tasks.register(it["task"])


def run_main(argv):
    import run as R
    sys.argv = argv
    R.main()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train"])
    p.add_argument("iteration")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--num_envs", type=int, required=True)
    p.add_argument("--resume_from", default="")
    a = p.parse_args()
    it = L.load(a.iteration)
    if it.get("imported"):
        sys.exit(f"{a.iteration} is an imported run ({it['imported'].get('from_run')}); it is not trained here")
    P = L.paths(a.iteration, a.seed)
    origin = prepare(it)
    print(f"[launch] {a.iteration} task={it['task']} ({origin}) run={P['run_name']} epochs={a.epochs} "
          f"envs={a.num_envs} seed={a.seed}" + (f" resume={L.rel(a.resume_from)}" if a.resume_from else ""), flush=True)
    os.environ["SMV2_RUN_NAME"] = P["run_name"]       # checkpoint names (skillmimic/run.py)
    argv = ["skillmimic/run.py"] + L.train_args(it) + [
        "--experiment", P["run_name"], "--num_envs", str(a.num_envs), "--max_epochs", str(a.epochs),
        "--seed", str(a.seed), "--output_path", P["runs_dir"] + "/", "--headless"]
    if a.resume_from:
        argv += ["--resume_from", a.resume_from]
    run_main(argv)


if __name__ == "__main__":
    main()
