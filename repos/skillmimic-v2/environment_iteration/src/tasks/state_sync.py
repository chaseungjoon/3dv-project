"""fix-8: reward, object termination and own-start capture read the post-step simulator state (U1 of
`TalkFile_3DV_Fix6_환경_업데이트_분석_및_실행계획.md` sections 5-6).

Why (fix-6 ITERATION.md section 3, verified again by src/probes/state_timing.py): the inherited
`SkillMimicParahome.post_physics_step` builds the reward input `_curr_obs` BEFORE `_refresh_sim_tensors()`. At control
step k+1 (progress_buf = k, action a_k just simulated) the order is

    _compute_hoi_observations()      _curr_obs   <- tensors of the previous refresh = state s_k
    _refresh_sim_tensors()           tensors     <- state s_{k+1}
    _compute_observations()          obs_buf     <- s_{k+1};  _curr_ref_obs <- hoi_data_batch[k] = ref frame sframe+k
    _compute_reward()                r_{k+1}     =  sim(ref sframe+k, s_k)
    _compute_reset()                 fall: s_{k+1};  book > objTerminationDist: s_k
    progress_buf += 1
    own-start capture                filter on |book(s_k) - ref(sframe+k)|, but stores s_{k+1} labelled frame sframe+k

so (a) the reward of action a_k is the imitation of s_k, which a_k cannot influence (one step of delay, and the first
reward after a reset compares the reference with the previous episode's rigid bodies); (b) the two terminations look
at different states; (c) own starts restore s_{k+1} against reference frame sframe+k, one frame behind the state's
real reference, and were filtered on a different state than the one stored (fix-6 live probe: up to 7.4 cm apart).

SkillMimicParahomeIterStateSync = fix-6's SkillMimicParahomeIterOwnStart + one optional env.yaml key:

  stateSync: True     # default False: the class then behaves exactly like its parent

With it, at control step k+1:

    _refresh_sim_tensors()           (unchanged, upstream)
    _compute_observations()          (unchanged: obs_buf, history row, phase, _curr_ref_obs <- hoi_data_batch[k])
    snapshot                         _curr_obs <- s_{k+1} (same refreshed tensors as obs_buf)
                                     _curr_ref_obs <- reference frame sframe+k+1 (hoi_data_batch[k+1])
    _compute_reward()                r_{k+1} = sim(ref sframe+k+1, s_{k+1})     (upstream function)
    _compute_reset()                 fall and book termination both on s_{k+1}  (upstream / fix-1 functions)
    progress_buf += 1, _hist_obs <- _curr_obs (unchanged)
    own-start capture                filter, stored state and stored frame from the same snapshot (frame sframe+k+1)
    _curr_ref_obs <- hoi_data_batch[k] again (legacy value, see "unchanged" below)

The state <-> reference pairing of the reward is the same as before (the state after n actions against reference
frame sframe+n); what changes is that the reward and termination of step k+1 are computed on s_{k+1} itself.

Details that keep everything else fixed:
- dof velocities in the reward input are still zeroed for the first post-reset state s_1 (build_hoi_observations is
  called with progress_buf + 1, the index of the state it is given).
- the velocity-smoothness term compares with `_hist_obs`; at a reset its dof-velocity block is set to the reset state's
  dof velocities (what the legacy path had there, its s_0 row), instead of the previous episode's last state.
- the last state of an episode (index E = episode length) has no row in hoi_data_batch; its reference is taken from the
  motion file (frame sframe+E, or the clip's last frame when the episode ends at the clip end; one frame of hold).
- the fix-1 book termination keeps its `progress_buf > 1` guard, so it now first checks s_3 (legacy: s_2), the same
  first state as the fall termination.
- unchanged: policy observation (obs_buf, history rows, phase value), episode length, number of rewards per episode,
  ATS bookkeeping, buffer-node averaging, own-start probability / frames / filter radius / restore, reward weights,
  physics. Evaluation definition unchanged: outside post_physics_step `_curr_ref_obs` holds its legacy value, so the
  traces of src/rollout.py (protocol v1) read exactly what they read for every earlier iteration. Only `rew_buf` (the
  trace's `reward` field) follows the new timing.

Each ATS rate update prints one `[sync]` line: the legacy reward-input lag that the sync removed (book and root
displacement between s_k and s_{k+1}, mean / max over the interval), a check that the reward reference equals the
motion-file row of the recorded frame, how often the motion-file tail row was used, and a check that each own-start
capture's stored error equals the distance recomputed from the stored book state and stored frame.
"""
import torch

from env.tasks.skillmimic_parahome import build_hoi_observations
from tasks.own_start import SkillMimicParahomeIterOwnStart

OBJ_POS = 6 + 2 * 180          # book position in the hoi observation / reference row (dof_obs_size = 180)


class SkillMimicParahomeIterStateSync(SkillMimicParahomeIterOwnStart):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._sync = bool(cfg["env"].get("stateSync", False))
        self._sync_ready = False
        if self._sync and float(cfg["env"].get("state_switch_prob", 0.0) or 0.0) > 0:
            raise SystemExit("stateSync assumes hoi_data_batch rows are motion-file rows (state_switch_prob = 0)")
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        if not self._sync:
            return
        assert self._dof_obs_size == 180 and OBJ_POS == 6 + 2 * self._dof_obs_size
        n, dev = self.num_envs, self.device
        self._legacy_ref_obs = torch.zeros_like(self._curr_ref_obs)
        self._sync_ref_obs = torch.zeros_like(self._curr_ref_obs)
        self._sync_ref_frame = torch.zeros(n, dtype=torch.long, device=dev)   # motion frame of _sync_ref_obs
        self._ref_tail = torch.zeros_like(self._curr_ref_obs)                # motion row of state index E
        self._sync_stats_reset()
        all_ids = torch.arange(n, device=dev)
        self._sync_after_reset(all_ids)
        self._sync_ready = True
        print(f"[sync] stateSync on: reward / termination / own-start capture on the post-step state | "
              f"termination height effective {float(self._termination_heights):.3f} m "
              f"(env.yaml terminationHeight {cfg['env'].get('terminationHeight')}, not read upstream) | "
              f"episode length {self.max_episode_length} | test={self.isTest}", flush=True)

    # ---------------------------------------------------------------- per-step snapshot
    def _compute_hoi_observations(self, env_ids=None):
        # Only caller: SkillMimicParahome.post_physics_step, before the refresh. With stateSync the reward input is
        # built after the refresh instead (_sync_snapshot).
        if self._sync and env_ids is None:
            return
        return super()._compute_hoi_observations(env_ids)

    def _sync_snapshot(self):
        md = self._motion_data
        prev_obj = self._hist_obs[:, OBJ_POS:OBJ_POS + 3].clone()       # = book of s_k (last step's _curr_obs)
        prev_root = self._hist_obs[:, 0:3].clone()
        self._legacy_ref_obs.copy_(self._curr_ref_obs)                   # hoi_data_batch[k], written by obs
        idx = self.progress_buf.long() + 1                               # index of the state s_{k+1}
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
                                                   self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:, 0, :],
                                                   self._dof_pos, self._dof_vel, key_body_pos,
                                                   self._local_root_obs, self._root_height_obs,
                                                   self._dof_obs_size, self._target_states,
                                                   self._hist_obs, idx)
        E = md.envid2episode_lengths.long()
        inside = idx < E
        rows = idx.clamp(max=self.hoi_data_batch.shape[1] - 1)
        ar = torch.arange(self.num_envs, device=self.device)
        ref = self.hoi_data_batch[ar, rows]
        ref = torch.where(inside.unsqueeze(-1), ref, self._ref_tail)
        self._sync_ref_obs.copy_(ref)
        self._curr_ref_obs.copy_(ref)
        L = md.motion_lengths[md.envid2motid].long()
        self._sync_ref_frame[:] = torch.minimum(md.envid2sframe.long() + torch.minimum(idx, E), L - 1)
        # bookkeeping for the [sync] line (GPU only, no host sync per step)
        if not self.isTest:
            s = self._sync_stats
            valid = idx >= 2                                             # s_k was a real post-step state
            lag_obj = (self._curr_obs[:, OBJ_POS:OBJ_POS + 3] - prev_obj).norm(dim=-1)
            lag_root = (self._curr_obs[:, 0:3] - prev_root).norm(dim=-1)
            s["n"] += valid.sum()
            s["lag_obj_sum"] += torch.where(valid, lag_obj, torch.zeros_like(lag_obj)).sum()
            s["lag_root_sum"] += torch.where(valid, lag_root, torch.zeros_like(lag_root)).sum()
            s["lag_obj_max"] = torch.maximum(s["lag_obj_max"], torch.where(valid, lag_obj, 0 * lag_obj).max())
            s["tail"] += (~inside).sum()
            s["steps"] += 1

    def _compute_reward(self):
        if self._sync:
            self._sync_snapshot()
        super()._compute_reward()

    def post_physics_step(self):
        super().post_physics_step()             # reward, reset, progress, _hist_obs, own-start capture
        if self._sync:
            self._curr_ref_obs.copy_(self._legacy_ref_obs)    # protocol-v1 meaning for rollout traces

    # ---------------------------------------------------------------- resets
    def _sync_after_reset(self, env_ids):
        md = self._motion_data
        D = self._dof_obs_size
        mids = md.envid2motid[env_ids].long()
        tail_frame = torch.minimum(md.envid2sframe[env_ids].long() + md.envid2episode_lengths[env_ids].long(),
                                   md.motion_lengths[mids].long() - 1)
        for m in torch.unique(mids).tolist():
            sel = mids == m
            self._ref_tail[env_ids[sel]] = md.hoi_data_dict[m]["hoi_data"][tail_frame[sel]].to(self._ref_tail.dtype)
        # velocity-smoothness reference of s_1 = the reset state's dof velocities (legacy: its s_0 reward input)
        self._hist_obs[env_ids, 6 + D:6 + 2 * D] = self._dof_vel[env_ids]

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self._sync and self._sync_ready and len(env_ids) > 0:
            self._sync_after_reset(torch.as_tensor(env_ids, device=self.device).long())

    # ---------------------------------------------------------------- own-start capture on the same snapshot
    def _own_capture(self):
        if not self._sync:
            return super()._own_capture()
        md = self._motion_data
        f = self._sync_ref_frame                       # motion frame the reward compared this state with
        err = (self._curr_obs[:, OBJ_POS:OBJ_POS + 3] - self._sync_ref_obs[:, OBJ_POS:OBJ_POS + 3]).norm(dim=-1)
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
        # self-check: stored book state vs the motion-file book at the stored frame == the filter's error
        mot, fr = self._ob_motion[rows], self._ob_frame[rows]
        ref_obj = torch.zeros(k, 3, device=self.device, dtype=self._ob_obj.dtype)
        for m in torch.unique(mot).tolist():
            sel = mot == m
            ref_obj[sel] = md.hoi_data_dict[m]["hoi_data"][fr[sel], OBJ_POS:OBJ_POS + 3].to(ref_obj.dtype)
        re = (self._ob_obj[rows, 0:3] - ref_obj).norm(dim=-1)
        s = self._sync_stats
        s["cap_check"] = torch.maximum(s["cap_check"], (re - self._ob_err[rows]).abs().max())

    # ---------------------------------------------------------------- logging
    def _sync_stats_reset(self):
        z = lambda: torch.zeros((), device=self.device)  # noqa: E731
        self._sync_stats = dict(n=z(), lag_obj_sum=z(), lag_root_sum=z(), lag_obj_max=z(), tail=z(), steps=0,
                                cap_check=z())

    def _sync_ref_check(self):
        """max |reward reference - motion-file row of the recorded frame| over envs whose row came from
        hoi_data_batch or the tail row (expected 0)."""
        md = self._motion_data
        worst = 0.0
        for m in torch.unique(md.envid2motid).tolist():
            sel = md.envid2motid == m
            rows = md.hoi_data_dict[m]["hoi_data"][self._sync_ref_frame[sel]]
            worst = max(worst, float((self._sync_ref_obs[sel] - rows).abs().max()))
        return worst

    def _reweight_motion(self, reset_env_ids):
        super()._reweight_motion(reset_env_ids)
        if not (self._sync and self._sync_ready) or self.isTest:
            return
        if not (self.progress_buf_total % self.reweight_intervel == 0 and self.progress_buf_total > 0):
            return
        s = self._sync_stats
        n = max(float(s["n"]), 1.0)
        print(f"[sync] step {int(self.progress_buf_total)}: legacy reward-input lag removed: book "
              f"{float(s['lag_obj_sum']) / n * 100:.2f} cm mean / {float(s['lag_obj_max']) * 100:.1f} cm max, root "
              f"{float(s['lag_root_sum']) / n * 100:.2f} cm mean | reference row check {self._sync_ref_check():.1e} "
              f"| tail rows {int(s['tail'])} in {s['steps']} steps | own capture error check "
              f"{float(s['cap_check']):.1e} m", flush=True)
        self._sync_stats_reset()
