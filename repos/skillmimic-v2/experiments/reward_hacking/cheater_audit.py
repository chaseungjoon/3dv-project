"""Test 3, cheater audit: how much training reward do hand-made "cheating" behaviours earn? (README.md)

No policy and no physics: each cheat is a fake state sequence built from the reference clip (e.g. the body moves
exactly like the demonstration but the object stays on the table). Each fake frame t is scored against reference
frame t with the training reward (experiments/tools/rh_reward.py, checked against the upstream jit function).
The environment is only built (1 env, about 1 min) to load the reference exactly as training sees it
(key bodies, dof layout, velocities, reward weights).

    python experiments/reward_hacking/cheater_audit.py --clip place_book
Output: experiments/results/reward_hacking/t3/<clip>_cheats.csv (+ <clip>_cheats.npz, per-frame reward and factors)

Columns
  fails_task      1 if the object is ever more than 10 cm from its reference (T09 task_success would be 0)
  clip_mean       mean per-frame reward over the clip (frames 2..end)
  train_return    expected training episode return: start frame uniform over 2..L-2 (as without ATS), at most
                  60 steps, ended once the object is > 0.25 m from its reference (fix env objTerminationDist)
  train_return_f2 the same episode started at frame 2 only (the evaluation start)
  *_vs_demo       ratio to the "demo" row (the demonstration itself)
A cheat that fails the task but keeps most of the demo's reward means the reward can be gamed by that behaviour.
"""
import argparse
import csv
import glob
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "tools"))
from isaacgym import gymapi  # noqa: E402,F401  (Isaac Gym must be imported before torch)
import methods as M  # noqa: E402
import rh_reward as RH  # noqa: E402

TASK_EPS_M, TERM_M, EP_LEN = 0.10, 0.25, 60


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, choices=["place_book", "drink_cup"])
    p.add_argument("--method", default="fix")
    p.add_argument("--out_dir", default="experiments/results/reward_hacking/t3")
    return p.parse_args()


def build_task(clip, method):
    cks = sorted(glob.glob(os.path.join(REPO, "experiments", "runs", f"{clip}_{method}_n2048_s0", "nn", "*_0*.pth")))
    if not cks:
        sys.exit(f"[t3] no checkpoint of {clip}_{method}_n2048_s0 (only needed to start the player, weights unused)")
    sys.argv = ["skillmimic/run.py", "--test"] + M.base_args(clip, method) + [
        "--num_envs", "1", "--episode_length", "300", "--state_init", "2", "--checkpoint", cks[-1], "--headless"]
    sys.path.insert(0, os.path.join(REPO, "skillmimic"))
    from rl_games.torch_runner import Runner
    import run as R
    captured = {}
    Runner.run = lambda self, args: captured.setdefault("runner", self)
    R.main()
    return captured["runner"].create_player().env.task


def main():
    a = parse()
    os.chdir(REPO)
    task = build_task(a.clip, a.method)
    import numpy as np
    import torch

    md = task._motion_data
    ref = md.hoi_data_dict[0]["hoi_data"].float()          # (L, D) exactly the rows the reward compares against
    L, D = ref.shape
    K, dof = len(task._key_body_ids), task._dof_obs_size
    lay = RH.layout(dof, K)
    assert lay["size"] == D, f"hoi layout {lay['size']} != reference width {D}"
    dev = ref.device
    w = {k: torch.full((L,), float(v), device=dev) for k, v in md.reward_weights_default.items()}
    OP, OR, OV, KP = lay["obj_pos"], lay["obj_rot"], lay["obj_vel"], lay["key_pos"]
    lift = (ref[:, OP][:, 2] - ref[2, OP][2]).cpu().numpy()
    lift_start = int(np.argmax(lift > 0.02)) if (lift > 0.02).any() else L // 3

    def demo():
        return ref.clone()

    def obj_static():                   # body exactly as demonstrated, object never touched
        o = ref.clone(); o[:, OP] = ref[2, OP]; o[:, OR] = ref[2, OR]; o[:, OV] = 0
        return o

    def obj_stops_at(x):                # carries the object until frame x, then leaves it there (stops carrying)
        def f():
            o = ref.clone(); o[x:, OP] = ref[x, OP]; o[x:, OR] = ref[x, OR]; o[x:, OV] = 0
            return o
        return f

    def late(k):                        # whole motion k frames behind the demonstration (timing lag)
        def f():
            idx = torch.clamp(torch.arange(L, device=dev) - k, min=2)
            return ref[idx].clone()
        return f

    def obj_late(k):                    # body on time, object k frames behind (late pick-up / late put-down)
        def f():
            idx = torch.clamp(torch.arange(L, device=dev) - k, min=2)
            o = ref.clone(); o[:, OP], o[:, OR], o[:, OV] = ref[idx][:, OP], ref[idx][:, OR], ref[idx][:, OV]
            return o
        return f

    def frozen():                       # stands still in the frame-2 pose, object untouched
        o = ref[2:3].repeat(L, 1).clone(); o[:, lay["dof_vel"]] = 0; o[:, OV] = 0
        return o

    def body_shift(d):                  # whole body d metres off (x), object follows the reference
        def f():
            o = ref.clone(); o[:, 0] += d
            kp = o[:, KP].view(L, K, 3); kp[..., 0] += d; o[:, KP] = kp.view(L, -1)
            return o
        return f

    def obj_shift(d):                   # body exact, object d metres off (x) the whole clip
        def f():
            o = ref.clone(); o[:, OP.start] += d
            return o
        return f

    xs = sorted({min(max(x, 3), L - 2) for x in (lift_start, lift_start + 15, lift_start + 30, lift_start + 60)})
    cheats = [("demo", demo), ("obj_static", obj_static), ("frozen", frozen)]
    cheats += [(f"obj_stops_at_{x}", obj_stops_at(x)) for x in xs]
    cheats += [(f"late_{k}", late(k)) for k in (5, 10, 20)]
    cheats += [(f"obj_late_{k}", obj_late(k)) for k in (5, 10, 20)]
    cheats += [(f"obj_shift_{int(d * 100)}cm", obj_shift(d)) for d in (0.05, 0.15, 0.25, 0.50)]
    cheats += [(f"body_shift_{int(d * 100)}cm", body_shift(d)) for d in (0.05, 0.10, 0.25)]

    owner = next(c for c in type(task).__mro__ if c.__name__ == "SkillMimicParahome")
    upstream = sys.modules[owner.__module__].compute_humanoid_reward
    zeros_c = torch.zeros((L, 1, 3), device=dev)

    rows, per_frame = [], {}
    for name, fn in cheats:
        obs = fn()
        hist = torch.cat([obs[:1], obs[:-1]], 0)
        parts = RH.reward_parts(ref, obs, hist, K, dof, w)
        up = upstream(ref, obs, hist, zeros_c, zeros_c[:, 0], K, dof, w)
        assert float((up - parts["rew"]).abs().max()) < 1e-4, f"{name}: rh_reward != upstream reward"
        rew = parts["rew"].cpu().numpy()
        err = (obs[:, OP] - ref[:, OP]).norm(dim=-1).cpu().numpy()
        rets = []
        for s in range(2, L - 1):                       # sample_time: start frames 2..L-2
            r_ = 0.0
            for k in range(min(L - s, EP_LEN)):
                r_ += float(rew[s + k])
                if k >= 1 and err[s + k] > TERM_M:
                    break
            rets.append(r_)
        row = dict(cheat=name, fails_task=int((err[2:] > TASK_EPS_M).any()), max_obj_err_m=round(float(err[2:].max()), 3),
                   clip_mean=float(rew[2:].mean()), clip_sum=float(rew[2:].sum()),
                   train_return=float(np.mean(rets)), train_return_f2=rets[0])
        for k in ("r_body", "r_obj", "r_ig"):
            row[k + "_mean"] = float(parts[k][2:].mean())
        rows.append(row)
        per_frame[name] = np.stack([parts[k].cpu().numpy() for k in RH.PARTS] + [err], 0)
    base = rows[0]
    for r in rows:
        for k in ("clip_mean", "train_return", "train_return_f2"):
            r[k + "_vs_demo"] = r[k] / base[k] if base[k] else float("nan")

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{a.clip}_cheats.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(path[:-4] + ".npz", names=np.array(list(per_frame)), parts=np.array(RH.PARTS + ["obj_err"]),
                        data=np.stack(list(per_frame.values())), lift_start=lift_start, clip_frames=L)
    print(f"[t3] {a.clip}: L={L} frames, reference lift starts at frame {lift_start}. Saved {path}")
    print(f"{'cheat':<22}{'fails':>6}{'clip_mean':>10}{'train_ret':>10}{'vs_demo':>9}{'f2_ret':>8}{'vs_demo':>9}")
    for r in rows:
        print(f"{r['cheat']:<22}{r['fails_task']:>6}{r['clip_mean']:>10.3f}{r['train_return']:>10.2f}"
              f"{r['train_return_vs_demo']:>9.2f}{r['train_return_f2']:>8.2f}{r['train_return_f2_vs_demo']:>9.2f}")


if __name__ == "__main__":
    main()
