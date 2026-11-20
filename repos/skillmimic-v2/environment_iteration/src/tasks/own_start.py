"""fix-5: training starts from the policy's own post-pickup states ("connection" starts, ENV_FIX_PLAN.md section 7).

Why (fix-4 ITERATION.md section 4-5, fix-5 ITERATION.md section 1): started from the reference state at frame >= 70
(book already up in the left hand) the fix-4 policy walks the carry (0.78 m along the reference walk, 2-3 steps of
~0.3 m), but every start that makes the policy do its own lift (reference frames 2 .. 65) ends in a wide-stance squat
(pelvis drop at frames 63-81) with both feet planted through the walk window (0 lift-offs from frame 2). Training
starts are always reference states, so the policy never practises the walk from the state its own pickup produces;
the reward along that path after frame 70 is ~0.

SkillMimicParahomeIterOwnStart = fix-4's SkillMimicParahomeIterATS + optional env.yaml keys (all off by default, then
the class behaves exactly like its parent):

  ownStartProb: 0.2            # fraction of training resets that start from a buffered own state (0 = off)
  ownStartFrames: [75, 95]     # reference frames at which own states are captured (inclusive)
  ownStartMaxStart: 60         # only from episodes that started at reference frame <= this (the policy did the lift)
  ownStartObjDist: 0.20        # only while the book is within this distance of its reference (m)
  ownStartBuffer: 1024         # ring buffer size (states); oldest are overwritten, so it follows the current policy
  ownStartPerStep: 8           # at most this many captures per control step
  ownStartMinFill: 64          # own starts begin once the buffer holds this many states

Capture (after every training step): for eligible envs (a reference-start episode, not terminated this step, not
fallen, conditions above) the simulator state is copied: humanoid root state, DOF position / velocity, book root
state, the 60-row history-encoder buffer, and the reference frame the state is matched against
(envid2sframe + progress - 1, the frame the reward compared it with).

Own start (at reset, after the normal RIS / STF / ATS start was drawn): with probability ownStartProb an env is
re-initialised from a random buffered state. Its reference is rebuilt from the captured frame f with upstream
get_initial_state (hoi_data_batch, episode length, phase, start frame), then the reference body / book state is
replaced by the captured one, and the history buffer by the captured history. The rest of the reset is upstream's
(including its use of the stale rigid-body tensor in the reset observation, as for every reset). Rewards, early
termination (fall, object > objTerminationDist) and PPO are unchanged.

ATS bookkeeping: an own-start episode's reward is NOT recorded into the per-start-frame ATS statistics (they keep
measuring reference starts only), and the ATS start histogram counts only reference starts.

Evaluation is unaffected: --test never captures or restores (rollout.py starts every env at frame 2). The buffer is
not saved in checkpoints; after a resume it refills in ~1 epoch (8 captures per step x 32 steps) before own starts
resume. Every rate update of ATS also prints one `[own]` line (buffer fill, own starts / captures since the last
line, captured frame range, median book error of the captured states).
"""
import numpy as np
import torch

from tasks.ats import SkillMimicParahomeIterATS


class SkillMimicParahomeIterOwnStart(SkillMimicParahomeIterATS):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        e = cfg["env"]
        self._own_p = float(e.get("ownStartProb", 0.0))
        self._own_frames = [int(x) for x in e.get("ownStartFrames", [75, 95])]
        self._own_max_start = int(e.get("ownStartMaxStart", 60))
        self._own_obj_dist = float(e.get("ownStartObjDist", 0.20))
        self._own_cap = int(e.get("ownStartBuffer", 1024))
        self._own_per_step = int(e.get("ownStartPerStep", 8))
        self._own_min_fill = int(e.get("ownStartMinFill", 64))
        self._own_ready = False
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        self._own_on = self._own_p > 0 and not self.isTest
        n, dev = self.num_envs, self.device
        self._own_flag = torch.zeros(n, dtype=torch.bool, device=dev)       # current episode is an own start
        self._own_pending = None                                             # (env_ids, buffer rows) of this reset
        if self._own_on:
            C = self._own_cap
            self._ob_root = torch.zeros(C, 13, device=dev)
            self._ob_dof_pos = torch.zeros(C, self.num_dof, device=dev)
            self._ob_dof_vel = torch.zeros(C, self.num_dof, device=dev)
            self._ob_obj = torch.zeros(C, 13, device=dev)
            self._ob_hist = torch.zeros(C, self.history_length, self.ref_hoi_data_size, device=dev)
            self._ob_frame = torch.zeros(C, dtype=torch.long, device=dev)
            self._ob_motion = torch.zeros(C, dtype=torch.long, device=dev)
            self._ob_err = torch.zeros(C, device=dev)
            self._ob_n = 0           # states written so far (fill = min(n, C))
            self._own_start_root = torch.zeros(n, 3, device=dev)      # restored root position (self-check)
            self._own_ref_root = torch.zeros(n, 3, device=dev)        # reference root at the restored frame
            self._own_stats_reset()
            print(f"[own] ownStart on: p={self._own_p} frames {self._own_frames} max start {self._own_max_start} "
                  f"obj dist {self._own_obj_dist} buffer {self._own_cap} per step {self._own_per_step} "
                  f"min fill {self._own_min_fill}", flush=True)
        self._own_ready = True

    # ---------------------------------------------------------------- capture
    def _own_stats_reset(self):
        self._own_stats = dict(starts=0, resets=0, captures=0, restore_err=0.0, step1_own=[], step1_ref=[],
                               rew_own=0.0, n_own=0, rew_ref=0.0, n_ref=0)

    def post_physics_step(self):
        super().post_physics_step()
        if self._own_ready and self._own_on:
            s = self._own_stats
            of = self._own_flag
            s["rew_own"] += float(self.rew_buf[of].sum()); s["n_own"] += int(of.sum())
            s["rew_ref"] += float(self.rew_buf[~of].sum()); s["n_ref"] += int((~of).sum())
            first = of & (self.progress_buf == 1)            # one step after an own start
            if first.any():
                root = self._humanoid_root_states[first, 0:3]
                s["step1_own"].append((root - self._own_start_root[first]).norm(dim=-1))
                s["step1_ref"].append((root - self._own_ref_root[first]).norm(dim=-1))
            self._own_capture()

    def _own_capture(self):
        md = self._motion_data
        f = md.envid2sframe + self.progress_buf - 1       # reference frame this state was compared with
        o = 6 + 2 * self._dof_obs_size                      # object position in the hoi observation
        err = (self._curr_obs[:, o:o + 3] - self._curr_ref_obs[:, o:o + 3]).norm(dim=-1)
        ok = (~self._own_flag) & (md.envid2sframe <= self._own_max_start) \
            & (f >= self._own_frames[0]) & (f <= self._own_frames[1]) \
            & (err < self._own_obj_dist) & (self._terminate_buf == 0) & (self.progress_buf > 1)
        ids = torch.nonzero(ok).flatten()
        if ids.numel() == 0:
            return
        if ids.numel() > self._own_per_step:
            ids = ids[torch.randperm(ids.numel(), device=self.device)[:self._own_per_step]]
        k = ids.numel()
        rows = (torch.arange(k, device=self.device) + self._ob_n) % self._own_cap
        self._ob_root[rows] = self._humanoid_root_states[ids]
        self._ob_dof_pos[rows] = self._dof_pos[ids]
        self._ob_dof_vel[rows] = self._dof_vel[ids]
        self._ob_obj[rows] = self._target_states[ids]
        self._ob_hist[rows] = self._hist_obs_batch[ids]
        self._ob_frame[rows] = f[ids]
        self._ob_motion[rows] = md.envid2motid[ids]
        self._ob_err[rows] = err[ids]
        self._ob_n += k
        self._own_stats["captures"] += k

    # ---------------------------------------------------------------- own start at reset
    def _reset_state_init(self, env_ids):
        super()._reset_state_init(env_ids)
        self._own_pending = None
        if not self._own_ready or len(env_ids) == 0:      # initial reset inside __init__
            return
        self._own_flag[env_ids] = False
        if not self._own_on:
            return
        fill = min(self._ob_n, self._own_cap)
        self._own_stats["resets"] += int(len(env_ids))
        if fill < self._own_min_fill:
            return
        sel = torch.rand(len(env_ids), device=self.device) < self._own_p
        inds = torch.nonzero(sel).flatten()
        if inds.numel() == 0:
            return
        rows = torch.randint(0, fill, (inds.numel(),), device=self.device)
        envs = env_ids[inds]
        md = self._motion_data
        mids = self._ob_motion[rows]
        frames = self._ob_frame[rows].to(torch.int)
        # reference rebuilt from the captured frame (hoi_data_batch, episode length, envid2sframe, reward weights)
        self.hoi_data_batch[envs], _, _, _, _, _, _, _, _, _, _ = md.get_initial_state(envs, mids, frames)
        mi, mt = self.motion_ids, self.motion_times                     # upstream keeps motion_ids on the CPU
        mi[inds.to(mi.device)] = mids.to(mi.device, mi.dtype)
        mt[inds.to(mt.device)] = frames.to(mt.device, mt.dtype)
        # the policy's own body / book state instead of the reference one
        r = self._ob_root[rows]
        self.init_root_pos[envs] = r[:, 0:3]
        self.init_root_rot[envs] = r[:, 3:7]
        self.init_root_pos_vel[envs] = r[:, 7:10]
        self.init_root_rot_vel[envs] = r[:, 10:13]
        self.init_dof_pos[envs] = self._ob_dof_pos[rows]
        self.init_dof_pos_vel[envs] = self._ob_dof_vel[rows]
        ob = self._ob_obj[rows]
        self.init_obj_pos[envs] = ob[:, 0:3]
        self.init_obj_rot[envs] = ob[:, 3:7]
        self.init_obj_pos_vel[envs] = ob[:, 7:10]
        self.init_obj_rot_vel[envs] = ob[:, 10:13]
        self.buffer_steps[envs] = 0          # not a noisy (STF) start: no buffer-node reward smoothing
        self._own_flag[envs] = True
        self._own_pending = (envs, inds, rows)
        self._own_stats["starts"] += int(inds.numel())

    def after_reset_actors(self, env_ids):
        pend = self._own_pending
        if pend is None:
            return super().after_reset_actors(env_ids)
        envs, inds, rows = pend
        # ATS start histogram (parent) counts reference starts only: hide own starts from it for this call
        hist_backup = self._ats_starts
        if hist_backup is not None:
            keep = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            keep[inds] = False
            self._ats_starts = None
        super().after_reset_actors(env_ids)
        if hist_backup is not None:
            self._ats_starts = hist_backup
            ids_k = torch.nonzero(keep).flatten().cpu()
            mids = self.motion_ids.cpu()[ids_k].detach().cpu().numpy().astype(np.int64).reshape(-1)
            mts = self.motion_times.cpu()[ids_k].detach().cpu().numpy().astype(np.int64).reshape(-1)
            for m in np.unique(mids):
                h = self._ats_starts.get(int(m))
                if h is not None:
                    np.add.at(h, np.clip(mts[mids == m], 0, len(h) - 1), 1)
        # history the reset observation sees = captured rows before the current one (oldest row repeated once);
        # _reset_envs puts the captured current row back after the reset observation appended the reference row
        H = self._ob_hist[rows]
        self._hist_obs_batch[envs] = torch.cat([H[:, :1], H[:, :-1]], dim=1)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        pend = self._own_pending
        if pend is not None:
            envs, _, rows = pend
            self._hist_obs_batch[envs] = self._ob_hist[rows]
            # self-check: the simulator now holds the buffered state (not the reference one)
            e1 = (self._humanoid_root_states[envs] - self._ob_root[rows]).abs().max()
            e2 = (self._dof_pos[envs] - self._ob_dof_pos[rows]).abs().max()
            e3 = (self._target_states[envs] - self._ob_obj[rows]).abs().max()
            self._own_stats["restore_err"] = max(self._own_stats["restore_err"], float(max(e1, e2, e3)))
            self._own_start_root[envs] = self._ob_root[rows, 0:3]
            self._own_ref_root[envs] = self.hoi_data_batch[envs, 0, 0:3]
            self._own_pending = None

    # ---------------------------------------------------------------- ATS bookkeeping
    def record_motion_time_reward(self, reset_env_ids):
        if not self._own_on:
            return super().record_motion_time_reward(reset_env_ids)
        if reset_env_ids is None:
            ids = torch.zeros(0, dtype=torch.long, device=self.device)
        else:
            ids = torch.as_tensor(reset_env_ids, device=self.device).reshape(-1).long()
        own = ids[self._own_flag[ids]] if ids.numel() else ids
        ref = ids[~self._own_flag[ids]] if ids.numel() else ids
        super().record_motion_time_reward(ref)
        if own.numel():
            for env_id in own.tolist():      # super wrote this step's reward as a running step; drop the episode
                self.envs_reward[env_id] = torch.zeros(self.max_episode_length, device=self.device)

    def _reweight_motion(self, reset_env_ids):
        super()._reweight_motion(reset_env_ids)
        if not (self._own_on and self.reweight):
            return
        if not (self.progress_buf_total % self.reweight_intervel == 0 and self.progress_buf_total > 0):
            return
        fill = min(self._ob_n, self._own_cap)
        s = self._own_stats
        if fill:
            fr = self._ob_frame[:fill].detach().cpu().numpy()
            er = self._ob_err[:fill].detach().cpu().numpy()
            rng = f"frames {fr.min()}-{fr.max()} (median {np.median(fr):.0f}) book err median {np.median(er):.3f} m"
        else:
            rng = "empty"
        d1 = torch.cat(s["step1_own"]).median().item() if s["step1_own"] else float("nan")
        d2 = torch.cat(s["step1_ref"]).median().item() if s["step1_ref"] else float("nan")
        print(f"[own] step {int(self.progress_buf_total)}: buffer {fill}/{self._own_cap} | own starts {s['starts']} "
              f"of {s['resets']} resets ({s['starts'] / max(s['resets'], 1) * 100:.1f}%) | captures {s['captures']} "
              f"| buffer {rng} | reward/step own {s['rew_own'] / max(s['n_own'], 1):.3f} ref "
              f"{s['rew_ref'] / max(s['n_ref'], 1):.3f} | check: restore max err {s['restore_err']:.2e}, root after "
              f"1 step {d1:.3f} m from restored / {d2:.3f} m from reference", flush=True)
        self._own_stats_reset()
