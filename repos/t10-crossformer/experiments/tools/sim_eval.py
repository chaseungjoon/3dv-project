"""Closed-loop evaluation of the frozen CrossFormer in SimplerEnv (visual matching), using SimplerEnv's own
evaluator and the official episode grids (tools/sim_suites.py).

    python experiments/tools/sim_eval.py --suite bridge --variant sim_baseline
    python experiments/tools/sim_eval.py --suite coke_can --variant sim_baseline --max-episodes-per-job 2  # check

Writes experiments/results/sim/<variant>/<suite>.json (success per episode and per task) and keeps videos,
per-step action logs and the evaluator's stdout in experiments/runs/sim/<variant>/<suite>/.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXP, RESULTS, RUNS, hide_tf_gpu  # noqa: E402
from evaluate import GpuMonitor  # noqa: E402
from sim_suites import EMBODIMENT, SUITES  # noqa: E402

ASSET_DIR = Path(os.environ.get("MS2_REAL2SIM_ASSET_DIR", EXP / "third_party/ManiSkill2_real2sim/data"))
os.environ["MS2_REAL2SIM_ASSET_DIR"] = str(ASSET_DIR)


def episode_stats(video_path):
    """SimplerEnv encodes the episode stats in the video name, e.g.
    failure_obj_episode_3_moved_correct_obj_False_..._src_on_target_False.mp4 or ..._n_lift_significant_0_...mp4"""
    name = Path(video_path).stem
    out = {k: v == "True" for k, v in re.findall(r"_([a-z][a-z_]*?)_(True|False)(?=_|$)", name)}
    for k, v in re.findall(r"_(n_[a-z_]+?)_(\d+)(?=_|$)", name):
        out[k] = int(v)
    out["success"] = name.startswith("success")
    return out


def stage_rates(stats):
    """Fraction of episodes in which each stage flag was ever true (numeric counters: > 0)."""
    keys = sorted({k for s in stats for k in s})
    return {k: float(np.mean([bool(s.get(k, False)) for s in stats])) for k in keys}


def limit_episodes(args, n):
    """Keep only the first n episodes of a job (pipeline check)."""
    if args.obj_variation_mode == "episode":
        lo, hi = args.obj_episode_range
        args.obj_episode_range = [lo, min(hi, lo + n)]
    else:
        args.obj_init_xs, args.obj_init_ys = args.obj_init_xs[:1], args.obj_init_ys[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--variant", default="sim_baseline")
    ap.add_argument("--policy", default="crossformer", choices=["crossformer", "octo-base", "octo-small"],
                    help="octo-*: SimplerEnv's own Octo wrapper and checkpoint (simulator-validity control)")
    ap.add_argument("--octo-init-rng", type=int, default=0)
    ap.add_argument("--max-episodes-per-job", type=int, default=None)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--checkpoint", default=str(EXP / "checkpoints/crossformer"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if not (ASSET_DIR / "real_inpainting").exists():
        sys.exit(f"SimplerEnv assets missing at {ASSET_DIR}: run experiments/scripts/setup_sim.sh")
    if a.policy == "crossformer":
        variant = yaml.safe_load(open(EXP / "configs/sim_variants.yaml"))[a.variant]
        run_name = a.variant + a.tag
    else:
        variant = {"policy": a.policy, "init_rng": a.octo_init_rng, "wrapper": "simpler_env OctoInference (official)"}
        run_name = f"{a.policy}_rng{a.octo_init_rng}{a.tag}"
    run_dir = RUNS / "sim" / run_name / a.suite
    run_dir.mkdir(parents=True, exist_ok=True)
    res_path = RESULTS / "sim" / run_name / f"{a.suite}.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["DISPLAY"] = ""  # headless rendering, as in SimplerEnv's main_inference.py
    hide_tf_gpu()
    from simpler_env.evaluation.argparse import get_args
    from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

    from crossformer.model.crossformer_model import CrossFormerModel
    from sim_policy import CrossFormerSimPolicy

    mon = GpuMonitor()
    mon.start()
    if a.policy == "crossformer":
        model = CrossFormerModel.load_pretrained(a.checkpoint)
    else:
        from octo.model.octo_model import OctoModel
        from simpler_env.policies.octo.octo_model import OctoInference

        model = OctoModel.load_pretrained(f"hf://rail-berkeley/{a.policy}")
    jobs = SUITES[a.suite](str(ASSET_DIR))
    if a.max_jobs:
        jobs = jobs[: a.max_jobs]

    results, t0 = [], time.time()
    for j, (task, job_args) in enumerate(jobs):
        sys.argv = ["sim_eval", "--policy-model", "crossformer", "--ckpt-path", run_name,
                    "--logging-dir", str(run_dir / "videos")] + job_args
        args = get_args()
        if a.max_episodes_per_job:
            limit_episodes(args, a.max_episodes_per_job)
        if a.policy == "crossformer":
            policy = CrossFormerSimPolicy(model, args.policy_setup, variant)
        else:
            policy = OctoInference(model=model, policy_setup=args.policy_setup, init_rng=a.octo_init_rng)
        ts = time.time()
        before = set((run_dir / "videos").rglob("*.mp4"))
        with open(run_dir / "evaluator_stdout.log", "a") as log, contextlib.redirect_stdout(log):
            success = maniskill2_evaluator(policy, args)
        new_videos = sorted(set((run_dir / "videos").rglob("*.mp4")) - before)
        bk = args.additional_env_build_kwargs or {}
        results.append({
            "task": task,
            "env_name": args.env_name,
            "urdf_version": str(bk.get("urdf_version", "None")),
            "build_kwargs": {k: str(v) for k, v in bk.items()},
            "robot_init": [float(args.robot_init_xs[0]), float(args.robot_init_ys[0])],
            "success": [bool(s) for s in success],
            "episode_stats": [episode_stats(v) for v in new_videos],
            "seconds": round(time.time() - ts, 1),
        })
        rate = np.mean(success) if success else float("nan")
        print(f"[{run_name}/{a.suite}] job {j + 1}/{len(jobs)} {task} {args.env_name} "
              f"{bk.get('urdf_version', '')}: {np.sum(success)}/{len(success)} = {rate:.2f} "
              f"({time.time() - ts:.0f}s)", flush=True)

    # per-task success (episodes pooled over URDF / pose variations, as in SimplerEnv's visual matching)
    per_task = {}
    for r in results:
        per_task.setdefault(r["task"], []).extend(r["success"])
    per_task_stats = {}
    for r in results:
        per_task_stats.setdefault(r["task"], []).extend(r["episode_stats"])
    summary = {k: {"success_rate": float(np.mean(v)), "n": len(v), "stages": stage_rates(per_task_stats[k])}
               for k, v in per_task.items()}
    out = {
        "variant": run_name,
        "policy": a.policy,
        "config": variant,
        "suite": a.suite,
        "embodiment": EMBODIMENT[a.suite],
        "partial": bool(a.max_episodes_per_job or a.max_jobs),
        "per_task": summary,
        "mean_success": float(np.mean([v["success_rate"] for v in summary.values()])),
        "episodes": sum(len(r["success"]) for r in results),
        "minutes": round((time.time() - t0) / 60, 1),
        "peak_gpu_mem_mib": mon.peak,
        "jobs": results,
    }
    res_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"suite": a.suite, "per_task": summary, "mean_success": out["mean_success"],
                      "minutes": out["minutes"]}, indent=1))
    print(f"-> {res_path}")


if __name__ == "__main__":
    main()
