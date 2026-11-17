"""Isaac Gym rollout engine of the loop: build an iteration's environment once, roll out checkpoints, record a rich
per-frame trace, optionally render camera images. Used by src/evaluate.py (no cameras) and src/render.py (cameras).

Episode = exactly experiments/tools/evaluate.py: reset every env at reference frame `start_frame` (2), run `horizon`
control steps, no restart (fall-only termination in --test mode, the first fall ends that env's episode), det = mean
action, stoch = sampled action. Perturbed start = the paper's eps-NSR draw (object yaw +-45 deg, xy offset <= 10 cm,
fixed per env by the seed), same RandomState sequence as evaluate.py.

Quick look (one checkpoint, a few envs; prints metrics and failure modes, writes nothing unless --out):
    python environment_iteration/src/rollout.py fix-1 --epoch 3000 --num_envs 8 --mode stoch --seed 101
"""
import argparse
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)   # torch.load weights_only notice (trusted local checkpoints)

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import iterlib as L  # noqa: E402
import launch  # noqa: E402

WRIST_R, WRIST_L = 10, 33          # jRightWrist, jLeftWrist (asset body order; evaluate.py WRIST_IDS)
REF_OBJ_POS = slice(366, 369)      # object position in _curr_ref_obs (evaluate.py)
REF_OBJ_ROT = slice(369, 373)      # object rotation (xyzw) right after it
REF_KEY0 = 376                     # key body positions start (evaluate.py)


class Env:
    """One built environment (fixed num_envs / mode / horizon) that can roll out many checkpoints and seeds."""

    def __init__(self, name, num_envs, mode="stoch", horizon=300, start_frame=2, seed=0, cameras=None,
                 env_spacing=None, first_checkpoint=None, state_init_random_prob=None, overrides=None):
        self.it = dict(L.load(name), **(overrides or {}))   # overrides: probes only (e.g. a test asset)
        self.name, self.num_envs, self.mode, self.horizon, self.start_frame = name, num_envs, mode, horizon, start_frame
        self.cameras = cameras
        launch.prepare(self.it)
        from isaacgym import gymapi  # noqa: F401
        import torch
        from rl_games.torch_runner import Runner
        import utils.parse_task as PT
        self.torch = torch

        if first_checkpoint is None:
            raise SystemExit("Env needs a checkpoint to build the player (first_checkpoint)")
        task_cls = getattr(PT, self.it["task"])
        if cameras is not None:
            cameras.install(task_cls)
        if env_spacing is not None:
            orig_ce = task_cls._create_envs

            def _create_envs(self_, n, spacing, per_row, _o=orig_ce):
                return _o(self_, n, float(env_spacing), per_row)
            task_cls._create_envs = _create_envs

        argv = ["skillmimic/run.py", "--test"] + L.base_args(self.it) + [
            "--num_envs", str(num_envs), "--episode_length", str(horizon), "--state_init", str(start_frame),
            "--checkpoint", first_checkpoint, "--seed", str(seed), "--headless"]
        captured = {}
        orig_run = Runner.run
        Runner.run = lambda self_, args: captured.setdefault("runner", self_)   # stop run.py before it plays
        try:
            launch.run_main(argv)
        finally:
            Runner.run = orig_run
        runner = captured["runner"]
        runner.config["player"] = dict(runner.config.get("player", {}), determenistic=(mode == "det"))
        self.player = runner.create_player()
        self.task = task = self.player.env.task
        # Clean starts (protocol.json eval.state_init_random_prob, normally 0): the RIS reset reads this cfg value on
        # every reset, also in --test, so the training value (0.1) would start ~7-10% of envs from a noisy state.
        if state_init_random_prob is None:
            state_init_random_prob = L.protocol()["eval"].get("state_init_random_prob", 0.0)
        self.state_init_random_prob = float(state_init_random_prob)
        task.cfg["env"]["state_init_random_prob"] = self.state_init_random_prob
        md = self.md = task._motion_data
        assert md.num_motions == 1, "one clip per policy"
        self.clip_frames = int(md.motion_lengths[0])
        self.n_clip = self.clip_frames - start_frame
        ref = md.hoi_data_dict[0]
        self.obj_z0 = float(ref["obj_pos"][start_frame, 2])
        self.ref_lift = float((ref["obj_pos"][:, 2] - ref["obj_pos"][0, 2]).max())
        self.ref_obj_path = ref["obj_pos"].detach().cpu().numpy()             # (clip_frames, 3) motion frames
        self.ref_root_path = ref["root_pos"].detach().cpu().numpy()
        clip = self.it["clip"]
        self.hold_thr = 0.2 if clip != "place_pan" else 0.35
        self.paper_z, self.paper_need = (1.2, 30) if clip.startswith("drink") else (0.9, 60)
        self.key_ids = task._key_body_ids
        cb = task._contact_body_ids              # cfg contactBodies: 5 left fingertips, then 5 right
        self.hand_l, self.hand_r = cb[:len(cb) // 2], cb[len(cb) // 2:]
        self._perturb = None
        orig_init = md.get_initial_state
        from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul

        def initial_state(env_ids, motion_ids, start_frames, _o=orig_init):
            out = _o(env_ids, motion_ids, start_frames)
            if self._perturb is None:
                return out
            yaw, dxy = self._perturb
            out = list(out)
            ids = torch.as_tensor(env_ids, device=task.device).long()
            zz = torch.zeros_like(yaw[ids])
            out[7] = out[7].clone(); out[7][:, :2] += dxy[ids]                       # obj_pos
            out[9] = quat_mul(quat_from_euler_xyz(zz, zz, yaw[ids]), out[9])         # obj_rot (xyzw)
            return tuple(out)
        md.get_initial_state = initial_state
        if cameras is not None:
            cameras.attach(self)

    def meta(self):
        return dict(iteration=self.name, clip=self.it["clip"], mode=self.mode, num_envs=self.num_envs,
                    horizon=self.horizon, start_frame=self.start_frame, clip_frames=self.clip_frames,
                    n_clip=self.n_clip, obj_z0=self.obj_z0, ref_lift=self.ref_lift, hold_thr=self.hold_thr,
                    paper_z=self.paper_z, paper_need=self.paper_need,
                    state_init_random_prob=self.state_init_random_prob)

    def set_perturb(self, seed):
        """evaluate.py --perturb draw for this seed (None = unperturbed)."""
        import numpy as np
        torch = self.torch
        if seed is None:
            self._perturb = None
            return
        n, dev = self.num_envs, self.task.device
        rs = np.random.RandomState(seed)
        yaw = torch.tensor(rs.uniform(-np.pi / 4, np.pi / 4, n), dtype=torch.float, device=dev)
        rad, ang = rs.uniform(0, 0.10, n), rs.uniform(0, 2 * np.pi, n)
        dxy = torch.tensor(np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), dtype=torch.float, device=dev)
        self._perturb = (yaw, dxy)

    def rollout(self, checkpoint, seed, perturb=False, on_frame=None, progress=False):
        """One episode per env. Returns (trace dict of numpy arrays (T, E, ...), ckpt info)."""
        import numpy as np
        torch, task, player = self.torch, self.task, self.player
        self.set_perturb(seed if perturb else None)
        player.restore(checkpoint)
        st = torch.load(checkpoint, map_location="cpu")
        info = dict(checkpoint=L.rel(checkpoint), epoch=st.get("epoch"), frame=st.get("frame"))
        del st
        torch.manual_seed(seed); np.random.seed(seed % (2 ** 32)); random.seed(seed)
        obs = player.env_reset()
        player.get_batch_size(obs["obs"], 1)
        E = self.num_envs
        alive = torch.ones(E, dtype=torch.bool, device=task.device)
        fall_t = torch.full((E,), -1, dtype=torch.long, device=task.device)
        keys = ["obj", "ref_obj", "root", "wrist_d", "body_err", "alive", "obj_rot", "ref_obj_rot", "root_rot",
                "wrist_r", "wrist_l", "wrist_d_r", "wrist_d_l", "key_pos", "ref_key_pos", "obj_contact",
                "hand_contact_r", "hand_contact_l", "reward", "ref_frame"]
        S = {k: [] for k in keys}
        nk = len(self.key_ids)
        t0 = time.time()
        for t in range(self.horizon):
            action = player.get_action(obs, self.mode == "det")
            obs_raw, _, done, inf = player.env_step(player.env, action)
            obs = player.obs_to_torch(obs_raw)       # no env_reset(done): a fallen env is not restarted
            rb = task._rigid_body_pos
            obj = task._target_states[:, 0:3].clone()
            wr, wl = rb[:, WRIST_R, :], rb[:, WRIST_L, :]
            dr, dl = (wr - obj).norm(dim=-1), (wl - obj).norm(dim=-1)
            kp = rb[:, self.key_ids, :]
            refk = task._curr_ref_obs[:, REF_KEY0:REF_KEY0 + 3 * nk].view(E, nk, 3)
            fell_now = inf["terminate"].bool() & alive
            fall_t[fell_now] = t
            cf = task._contact_forces
            vals = dict(
                obj=obj, ref_obj=task._curr_ref_obs[:, REF_OBJ_POS].clone(),
                root=task._humanoid_root_states[:, 0:3].clone(), wrist_d=torch.minimum(dr, dl),
                body_err=(kp - refk).norm(dim=-1).mean(-1), alive=alive & ~fell_now,
                obj_rot=task._target_states[:, 3:7].clone(), ref_obj_rot=task._curr_ref_obs[:, REF_OBJ_ROT].clone(),
                root_rot=task._humanoid_root_states[:, 3:7].clone(), wrist_r=wr.clone(), wrist_l=wl.clone(),
                wrist_d_r=dr, wrist_d_l=dl, key_pos=kp.clone(), ref_key_pos=refk.clone(),
                obj_contact=task._tar_contact_forces.norm(dim=-1),
                hand_contact_r=cf[:, self.hand_r, :].norm(dim=-1).sum(-1),
                hand_contact_l=cf[:, self.hand_l, :].norm(dim=-1).sum(-1),
                reward=task.rew_buf.clone(), ref_frame=self.md.envid2sframe.to(task.device) + task.progress_buf)
            for k in keys:
                S[k].append(vals[k].detach().clone())
            if on_frame is not None:
                on_frame(t, {k: v for k, v in vals.items()})
            alive = alive & ~fell_now & ~(done.bool() & (t < self.horizon - 1))
            if progress and (t + 1) % 50 == 0:
                print(f"[rollout] step {t + 1}/{self.horizon} ({time.time() - t0:.0f}s)", flush=True)
        S = {k: torch.stack(v).cpu().numpy() for k, v in S.items()}
        S["fall_t"] = fall_t.cpu().numpy()
        S["alive"] = S["alive"].astype(bool)
        return S, info


def consts(env):
    m = env.meta()
    import metrics as MX
    return dict(n_clip=m["n_clip"], obj_z0=m["obj_z0"], ref_lift=m["ref_lift"], thr=m["hold_thr"],
                paper_z=m["paper_z"], paper_need=m["paper_need"],
                box=MX.load_box(L.REPO, env.it.get("object_box", MX.BOOK_BOX)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--seed_run", type=int, default=None, help="training seed of the run (default protocol seed)")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--mode", default="stoch", choices=["det", "stoch"])
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--out", default="", help="write the trace .npz here")
    a = p.parse_args()
    import numpy as np
    import metrics as MX
    P = L.paths(a.iteration, a.seed_run)
    ck = L.checkpoint_at(P["run_dir"], a.epoch)
    if not ck:
        sys.exit(f"no checkpoint for epoch {a.epoch} in {L.rel(P['run_dir'])}/nn")
    env = Env(a.iteration, a.num_envs, a.mode, a.horizon, seed=a.seed, first_checkpoint=ck)
    S, info = env.rollout(ck, a.seed, a.perturb, progress=True)
    c = consts(env)
    eps = [MX.episode_metrics(S, S["fall_t"], c, e) for e in range(a.num_envs)]
    ph = MX.ref_phases(MX.clean_ref(S, c["n_clip"]), c["obj_z0"])
    fe = [MX.failure_events(S, S["fall_t"], c, e, ph) for e in range(a.num_envs)]
    agg = MX.aggregate(eps)
    print(f"[rollout] {a.iteration} epoch {info['epoch']}: task_success={agg['task_success']:.2f} "
          f"lift={agg['lift_success']:.2f} paper={agg['paper_success']:.2f} progress={agg['task_progress']:.2f}")
    print(f"[rollout] reference phases (frames): {ph}")
    for f in fe:
        print(f"  env {f['env']:2d}: {f['mode']:9s} at frame {f['first_bad']:3d} ({f['phase']}) {f['detail']}")
    if a.out:
        np.savez_compressed(a.out, **S)
        print(f"[rollout] trace -> {a.out}")


if __name__ == "__main__":
    main()
