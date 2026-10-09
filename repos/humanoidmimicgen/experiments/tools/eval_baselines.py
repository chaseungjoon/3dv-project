"""Non-learned reference policies in the exact official evaluation loop (calibrates how hard the success test is).

The official evaluator (scripts/evaluate_policy_example.py) counts an episode as a success if the task
predicate is true at any step within 1250 steps (Push Button: button joint pressed >= 1 cm). The 2026-10-09
pipeline check found that a DP trained for only 200 updates (loss 0.97) succeeded 2/2, so the success
rate alone may say little about policy quality. These references give the floor:

  hold    : stand still (zero navigation, the first training frame's upper-body pose)
  mean    : the training-set mean action every step (walks slowly forward with an average arm pose)
  replay  : open-loop replay of a random training demonstration's actions (ignores the scene layout)
  noise   : uniform random actions inside the training action range (resampled every 50 steps)

    python experiments/tools/eval_baselines.py --policy mean --num-episodes 20 --output experiments/results/baselines/mean_20ep.json

Same seeds (0..N-1), same environment construction and same success function as the official evaluator.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import evaluate_policy_example as E  # noqa: E402


def load_actions(data_root: Path, task: str):
    import pyarrow.parquet as pq
    eps = []
    for shard in sorted(glob.glob(str(data_root / f"local/hmg_{task}_shard_*"))):
        for f in sorted(glob.glob(os.path.join(shard, "data", "*", "*.parquet"))):
            t = pq.read_table(f, columns=["episode_index", "action"]).to_pandas()
            for _, g in t.groupby("episode_index"):
                eps.append(np.stack(g["action"].to_numpy()).astype(np.float64))
    return eps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["hold", "mean", "replay", "noise"], required=True)
    ap.add_argument("--task", default="02_push_button", choices=sorted(E.TASKS))
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--max-episode-steps", type=int, default=1250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", type=Path, default=REPO / "experiments/data/policy_data/projected")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import humanoidmimicgen.locomanipulation  # noqa: F401
    from humanoidmimicgen.dataset_playback import ActionEnv, PlaybackConfig, get_task_success
    from humanoidmimicgen.wbc_runtime import get_env, get_policies, get_robot_type_and_model

    demos = load_actions(a.data_root, a.task)
    allact = np.concatenate(demos)
    lo, hi, mean = allact.min(0), allact.max(0), allact.mean(0)
    rng = np.random.RandomState(a.seed)
    print(f"[baseline] {a.policy}: {len(demos)} demos, {len(allact)} frames in the training set")

    config = PlaybackConfig(task_name=E.TASKS[a.task], enable_offscreen=True, enable_onscreen=False)
    robot_type, robot_model = get_robot_type_and_model(config.robot, config.enable_waist)
    sync_env = get_env(config, onscreen=False, offscreen=True, camera_names=["robot0_oak_egoview"],
                       camera_heights=[480], camera_widths=[640])
    successes, steps_l = [], []
    t0 = time.time()
    try:
        for ep in range(a.num_episodes):
            sync_env.reset(seed=a.seed + ep)
            wbc_policy, _, _ = get_policies(config, robot_type, robot_model, activate_keyboard_listener=False)
            env = ActionEnv(sync_env, wbc_policy)
            demo = demos[rng.randint(len(demos))]
            hold = demos[0][0].copy(); hold[E.UPPER_BODY_DIM:E.UPPER_BODY_DIM + E.NAVIGATION_DIM] = 0.0
            success = terminated = truncated = False
            steps = 0
            noise = mean
            while not (terminated or truncated) and steps < a.max_episode_steps:
                if a.policy == "hold":
                    act = hold
                elif a.policy == "mean":
                    act = mean
                elif a.policy == "replay":
                    act = demo[min(steps, len(demo) - 1)]
                else:
                    if steps % 50 == 0:
                        noise = lo + rng.rand(len(lo)) * (hi - lo)
                    act = noise
                _, _, terminated, truncated, _ = env.step(E.action_to_wbc_goal(act))
                success = success or get_task_success(sync_env)
                steps += 1
                if success:
                    break
            successes.append(bool(success)); steps_l.append(steps)
            print(f"[baseline {a.policy}] episode {ep + 1}/{a.num_episodes} success={success} steps={steps}", flush=True)
    finally:
        sync_env.close()
    res = dict(policy=a.policy, task_preset=a.task, num_episodes=a.num_episodes, seed=a.seed,
               max_episode_steps=a.max_episode_steps, episode_successes=successes, episode_steps=steps_l,
               success_rate=sum(successes) / len(successes), wall_s=round(time.time() - t0))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.output, "w"), indent=2)
    print(f"[baseline {a.policy}] success_rate={res['success_rate']:.2f} ({sum(successes)}/{len(successes)}) -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
