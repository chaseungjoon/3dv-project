"""Evaluate SkillMimic-V2 ParaHome checkpoints (experiments/docs/PROTOCOL.md 4).

Builds the environment once per (clip, method) and scores many checkpoints in one process.
One episode = start at reference frame 2 (as the upstream test commands, --state_init 2), run
`--horizon` control steps, no restart: the first fall ends the episode for that env (sticky).

    python experiments/tools/evaluate.py --run_dir experiments/runs/<run> --which all          # learning curve
    python experiments/tools/evaluate.py --run_dir experiments/runs/<run> --mode stoch         # final, 32 noisy rollouts
    python experiments/tools/evaluate.py --run_dir experiments/runs/<run> --perturb            # eps-neighbourhood start

Output: experiments/results/eval/<mode>[_perturb]/<run_name>/<ckpt>.json (+ per-frame .npz)

Metrics (per env, then averaged):
  task_success   T09-comparable: world object position within 10 cm of the reference at every clip
                 frame AND no fall during the clip (same rule as T09 `task_success`, BASELINE_T09.md 0)
  lift_success   the object was lifted to >= 50% of the reference lift while within 0.2 m of a wrist
  paper_success  upstream metric (skillmimic/metric/{place,drink}_metric.py) on this episode only:
                 Place: >60 frames with root z>0.5, wrist-object < thr, object z>0.9; Drink: >30 frames, z>1.2
  ...plus object/body tracking errors, fall frame, held frames.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import methods as M  # noqa: E402

TASK_EPS_M = 0.10
WRIST_IDS = [10, 33]  # right, left wrist (skillmimic_parahome.get_state_for_metric)
CKPT_EPOCH_RE = re.compile(r"_(\d{8})\.pth$")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", nargs="*", default=[])
    p.add_argument("--checkpoints", nargs="*", default=[], help="explicit .pth files (need --clip/--method)")
    p.add_argument("--clip", default="", help="override run_meta.json")
    p.add_argument("--method", default="", help="override run_meta.json")
    p.add_argument("--which", default="final", choices=["final", "all"])
    p.add_argument("--epochs", default="", help="comma-separated epochs (overrides --which)")
    p.add_argument("--mode", default="det", choices=["det", "stoch"])
    p.add_argument("--num_envs", type=int, default=0, help="default det 8, stoch 32")
    p.add_argument("--horizon", type=int, default=300, help="control steps per episode (30 Hz)")
    p.add_argument("--start_frame", type=int, default=2)
    p.add_argument("--perturb", action="store_true",
                   help="paper eps-NSR start: object yaw +-45 deg and xy offset <= 10 cm (fixed per env by --seed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="experiments/results/eval")
    p.add_argument("--tag", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--viewer", action="store_true", help="1 env with the Isaac Gym viewer")
    return p.parse_args()


def ckpt_epoch(path):
    m = CKPT_EPOCH_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def select_checkpoints(run_dir, which, epochs):
    nn = os.path.join(run_dir, "nn")
    numbered = sorted((ckpt_epoch(p), p) for p in glob.glob(os.path.join(nn, "*.pth")) if ckpt_epoch(p) is not None)
    if epochs:
        want = {int(e) for e in epochs.split(",") if e}
        return [p for e, p in numbered if e in want]
    if which == "all":
        return [p for _, p in numbered]
    return [numbered[-1][1]] if numbered else []


def run_meta(run_dir):
    meta = {"run_dir": os.path.relpath(run_dir, REPO), "run_name": os.path.basename(os.path.normpath(run_dir))}
    path = os.path.join(run_dir, "run_meta.json")
    if os.path.isfile(path):
        with open(path) as f:
            meta.update(json.load(f))
    return meta


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    a = parse()
    os.chdir(REPO)
    jobs = []
    for rd in a.run_dir:
        rd = os.path.abspath(rd)
        cks = select_checkpoints(rd, a.which, a.epochs)
        if not cks:
            print(f"[eval] no numbered checkpoints in {rd}/nn, skipped")
        jobs += [(run_meta(rd), c) for c in cks]
    for c in a.checkpoints:
        c = os.path.abspath(c)
        jobs.append((run_meta(os.path.dirname(os.path.dirname(c))), c))
    for meta, _ in jobs:
        meta["clip"] = a.clip or meta.get("clip")
        meta["method"] = a.method or meta.get("method")
    if not jobs:
        sys.exit("[eval] nothing to evaluate")
    keys = {(m["clip"], m["method"]) for m, _ in jobs}
    if len(keys) != 1 or None in next(iter(keys)):
        sys.exit(f"[eval] one (clip, method) per call, got {keys}; pass --clip/--method or split the call")
    clip, method = next(iter(keys))

    num_envs = 1 if a.viewer else (a.num_envs or (8 if a.mode == "det" else 32))
    sub = a.mode + ("_perturb" if a.perturb else "") + (f"_{a.tag}" if a.tag else "")
    out_root = os.path.join(a.out_dir, sub)

    def out_path(meta, ckpt):
        return os.path.join(out_root, meta["run_name"], os.path.splitext(os.path.basename(ckpt))[0] + ".json")

    if not a.force:
        todo = [(m, c) for m, c in jobs if not os.path.isfile(out_path(m, c))]
        if len(todo) < len(jobs):
            print(f"[eval] {len(jobs) - len(todo)} checkpoint(s) already evaluated, skipped (--force to redo)")
        jobs = todo
    if not jobs:
        print("[eval] all done")
        return

    # ---- same flags as a run.py --test call ----
    sys.argv = ["skillmimic/run.py", "--test"] + M.base_args(clip, method) + [
        "--num_envs", str(num_envs), "--episode_length", str(a.horizon), "--state_init", str(a.start_frame),
        "--checkpoint", jobs[0][1], "--seed", str(a.seed)]
    if not a.viewer:
        sys.argv.append("--headless")
    sys.path.insert(0, os.path.join(REPO, "skillmimic"))
    from isaacgym import gymapi  # noqa: F401  (before torch)
    import numpy as np
    import torch
    from rl_games.torch_runner import Runner
    import run as R

    captured = {}
    Runner.run = lambda self, args: captured.setdefault("runner", self)  # stop main() before it plays
    R.main()
    runner = captured["runner"]
    runner.config["player"] = dict(runner.config.get("player", {}), determenistic=(a.mode == "det"))
    player = runner.create_player()
    task = player.env.task
    mdata = task._motion_data
    assert mdata.num_motions == 1, "one clip per policy"
    clip_len = int(mdata.motion_lengths[0])
    n_clip = clip_len - a.start_frame  # control steps that still have a reference frame
    ref = mdata.hoi_data_dict[0]
    obj_z0 = float(ref["obj_pos"][a.start_frame, 2])
    ref_lift = float((ref["obj_pos"][:, 2] - ref["obj_pos"][0, 2]).max())
    thr = 0.2 if clip != "place_pan" else 0.35
    paper_z, paper_need = (1.2, 30) if clip.startswith("drink") else (0.9, 60)
    key_ids = task._key_body_ids

    if a.perturb:  # object-centric eps perturbation of the initial state, same draw for every checkpoint
        rs = np.random.RandomState(a.seed)
        yaw = torch.tensor(rs.uniform(-np.pi / 4, np.pi / 4, num_envs), dtype=torch.float, device=task.device)
        rad, ang = rs.uniform(0, 0.10, num_envs), rs.uniform(0, 2 * np.pi, num_envs)
        dxy = torch.tensor(np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), dtype=torch.float, device=task.device)
        from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul
        orig = mdata.get_initial_state

        def perturbed(env_ids, motion_ids, start_frames):
            out = list(orig(env_ids, motion_ids, start_frames))
            ids = torch.as_tensor(env_ids, device=task.device).long()
            z = torch.zeros_like(yaw[ids])
            out[7] = out[7].clone(); out[7][:, :2] += dxy[ids]                                   # obj_pos
            out[9] = quat_mul(quat_from_euler_xyz(z, z, yaw[ids]), out[9])                       # obj_rot (xyzw)
            return tuple(out)
        mdata.get_initial_state = perturbed

    commit = git_commit()
    print(f"[eval] env ready: {clip}/{method} envs={num_envs} mode={a.mode} horizon={a.horizon} "
          f"clip_frames={clip_len} perturb={a.perturb}")
    for k, (meta, ckpt) in enumerate(jobs):
        t0 = time.time()
        player.restore(ckpt)
        st = torch.load(ckpt, map_location="cpu")
        epoch, frame = st.get("epoch", ckpt_epoch(ckpt)), st.get("frame")
        del st
        torch.manual_seed(a.seed)
        obs = player.env_reset()                  # all envs at reference frame start_frame
        player.get_batch_size(obs["obs"], 1)

        alive = torch.ones(num_envs, dtype=torch.bool, device=task.device)
        fall_t = torch.full((num_envs,), -1, dtype=torch.long, device=task.device)
        S = {k_: [] for k_ in ["obj", "ref_obj", "root", "wrist_d", "body_err", "alive"]}
        for t in range(a.horizon):
            if a.viewer:
                p = task._humanoid_root_states[0, 0:3].cpu().numpy()
                task.gym.viewer_camera_look_at(task.viewer, None, gymapi.Vec3(p[0] + 1.5, p[1] - 1.5, 1.6),
                                               gymapi.Vec3(p[0], p[1], 0.9))
                task.render()
            action = player.get_action(obs, a.mode == "det")
            obs_raw, _, done, info = player.env_step(player.env, action)
            obs = player.obs_to_torch(obs_raw)       # no env_reset(done): fallen envs are not restarted
            # state after this step, before the auto-reset of finished envs on the next step
            obj = task._target_states[:, 0:3].clone()
            refo = task._curr_ref_obs[:, 366:369].clone()      # reference used by this step's reward
            wr = task._rigid_body_pos[:, WRIST_IDS, :]
            wd = (wr - obj[:, None]).norm(dim=-1).min(-1).values
            kp = task._rigid_body_pos[:, key_ids, :].reshape(num_envs, -1)
            refk = task._curr_ref_obs[:, 376:376 + 3 * len(key_ids)]
            be = (kp - refk).view(num_envs, -1, 3).norm(dim=-1).mean(-1)
            fell_now = info["terminate"].bool() & alive
            fall_t[fell_now] = t
            S["alive"].append(alive & ~fell_now)
            for key_, v in zip(["obj", "ref_obj", "root", "wrist_d", "body_err"],
                               [obj, refo, task._humanoid_root_states[:, 0:3].clone(), wd, be]):
                S[key_].append(v)
            alive = alive & ~fell_now & ~(done.bool() & (t < a.horizon - 1))
        S = {k_: torch.stack(v).cpu().numpy() for k_, v in S.items()}  # (T, envs, ...)

        eps = []
        T = a.horizon
        nc = min(n_clip, T)
        for e in range(num_envs):
            al = S["alive"][:, e]
            fell = bool(fall_t[e] >= 0)
            ft = int(fall_t[e]) if fell else -1
            fell_in_clip = fell and ft < nc
            err = np.linalg.norm(S["obj"][:nc, e] - S["ref_obj"][:nc, e], axis=-1)
            err_alive = err[al[:nc]] if al[:nc].any() else err[:1]
            bad = (err > TASK_EPS_M) | ~al[:nc]
            first_bad = int(np.argmax(bad)) if bad.any() else nc
            z = S["obj"][:, e, 2]
            near = S["wrist_d"][:, e] < thr
            held = al & near & (z - obj_z0 > 0.05)
            at_paper = al & (S["root"][:, e, 2] > 0.5) & near & (z > paper_z)
            lift_ok = bool((al & near & (z - obj_z0 >= 0.5 * ref_lift)).any())
            eps.append(dict(
                fell=int(fell), fall_frame=ft, fell_in_clip=int(fell_in_clip),
                task_success=int((not fell_in_clip) and float(err.max()) <= TASK_EPS_M),
                task_progress=first_bad / nc,
                obj_err_mean=float(err_alive.mean()), obj_err_max=float(err.max()),
                obj_err_end=float(err[nc - 1]) if al[nc - 1] else None,
                body_err_mean=float(S["body_err"][:nc, e][al[:nc]].mean()) if al[:nc].any() else None,
                lift_max=float((z[al] - obj_z0).max()) if al.any() else 0.0,
                lift_success=int(lift_ok),
                held_frames=int(held.sum()), held_frames_clip=int(held[:nc].sum()),
                paper_frames=int(at_paper.sum()), paper_success=int(at_paper.sum() > paper_need),
                paper_success_clip=int(at_paper[:nc].sum() > paper_need),
            ))
        agg = {k_: float(np.mean([e_[k_] for e_ in eps if e_[k_] is not None]))
               for k_ in eps[0] if any(e_[k_] is not None for e_ in eps)}
        agg["n_episodes"] = len(eps)
        res = dict(meta=dict(meta, checkpoint=os.path.relpath(ckpt, REPO), epoch=epoch, frame=frame, mode=a.mode,
                             num_envs=num_envs, horizon=a.horizon, start_frame=a.start_frame, perturb=a.perturb,
                             seed=a.seed, clip_frames=clip_len, ref_lift=ref_lift, task_eps_m=TASK_EPS_M,
                             git=commit, evaluated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                             eval_seconds=round(time.time() - t0, 1)),
                   aggregate=agg, episodes=eps)
        path = out_path(meta, ckpt)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        np.savez_compressed(path[:-5] + ".npz", **S)
        print(f"[eval {k + 1}/{len(jobs)}] {meta['run_name']} epoch={epoch} | task_success={agg['task_success']:.2f} "
              f"lift_success={agg['lift_success']:.2f} paper_success={agg['paper_success']:.2f} "
              f"obj_err={agg['obj_err_mean']:.3f}m fell_in_clip={agg['fell_in_clip']:.2f} "
              f"task_progress={agg['task_progress']:.2f} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
