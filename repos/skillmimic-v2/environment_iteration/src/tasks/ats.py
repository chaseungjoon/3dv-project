"""fix-4: adaptive trajectory sampling (ATS) that actually reaches the start-frame sampler.

Upstream bug (FIX1_ANALYSIS.md section 3, verified 2026-11-19): with `--reweight` the ParaHome task records the
per-start-frame reward and recomputes `ParaMotionDataHandler.time_sample_rate` every 5 x clip-length steps
(skillmimic/env/tasks/skillmimic_parahome.py `_reweight_motion`), but `_load_motion` builds the data handler without
`reweight=...`, so `ParaMotionDataHandler.reweight` stays False and `sample_time` always takes the uniform branch
(skillmimic/utils/paramotion_data_handler.py `sample_time`). Every earlier run (PAPER, FIX, fix-1..fix-3) therefore
trained with uniform start frames although the rates were computed. The upstream reweight branch of `sample_time`
is also broken (it raises TypeError on its first call, found when fix-4 first enabled it), so this class binds a
correct sampler to the handler (`_ats_sample_time`).

SkillMimicParahomeIterATS = fix-1's SkillMimicParahomeLocalHistRISBuffernodePhase + one optional env.yaml key:

  atsSampler: True     # hand the task's reweight flag (run.py --reweight) to the start-frame sampler

Without the key (or False) the class behaves exactly like its parent. With it, training starts are drawn from
time_sample_rate = (1 - alpha) / (L - 3) + alpha * softmax(-5 * mean reward of the start frame) (alpha =
--reweight_alpha, 1.0 in the loop), i.e. the frames the policy imitates worst are started from more often, as the
paper describes. Evaluation is unaffected: src/rollout.py starts every env at a fixed frame (--state_init) and test
runs do not pass --reweight.

Each rate update prints one `[ats]` line (step, the sampler's effective flag, max/min probability, entropy ratio,
the 5 most-sampled reference frames) and every 20th update a histogram of the actual start frames since the last
print, so the training log shows the sampler is active and where it concentrates.
"""
import types

import numpy as np
import torch

from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRISBuffernodePhase
from utils.paramotion_data_handler import ParaMotionDataHandler


def _ats_sample_time(md, motion_ids, truncate_time=None):
    """ParaMotionDataHandler.sample_time with a working reweight branch (bound to the handler instance).

    Upstream's reweight branch never ran and crashes when it does (`zip(start, end)` with the int start = 2:
    TypeError 'int' object is not iterable). Same support as the uniform branch: reference frames 2 .. L-2, i.e.
    L-3 values, the length of time_sample_rate[m] (index i = frame i + 2). The uniform branch is upstream's own.
    """
    if not md.reweight or md.play_dataset:
        return ParaMotionDataHandler.sample_time(md, motion_ids, truncate_time)
    ids = torch.as_tensor(motion_ids).detach().cpu().numpy().astype(np.int64).reshape(-1)
    times = np.empty(len(ids), dtype=np.int64)
    for m in np.unique(ids):
        p = torch.as_tensor(md.time_sample_rate[int(m)]).detach().cpu().double().numpy()
        n_frames = int(md.motion_lengths[int(m)].item()) - 3
        assert len(p) == n_frames and np.all(np.isfinite(p)) and p.sum() > 0, (len(p), n_frames, p.sum())
        sel = ids == m
        times[sel] = 2 + np.random.choice(n_frames, size=int(sel.sum()), p=p / p.sum())
    motion_times = torch.tensor(times, device=md.device, dtype=torch.int)
    if truncate_time is not None:
        assert truncate_time >= 0
        motion_times = torch.min(motion_times, md.motion_lengths[motion_ids] - truncate_time)
    return motion_times


class SkillMimicParahomeIterATS(SkillMimicParahomeLocalHistRISBuffernodePhase):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._ats_on = bool(cfg["env"].get("atsSampler", False))
        self._ats_updates = 0
        self._ats_starts = None
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

    def _load_motion(self, motion_file, dof_obs_size):
        super()._load_motion(motion_file, dof_obs_size)
        if self._ats_on:
            self._motion_data.reweight = bool(self.reweight)
            self._motion_data.sample_time = types.MethodType(_ats_sample_time, self._motion_data)
            L = {int(k): int(v) for k, v in
                 ((m, self._motion_data.motion_lengths[m].item()) for m in self._motion_data.hoi_data_dict)}
            self._ats_starts = {m: np.zeros(n, dtype=np.int64) for m, n in L.items()}
            print(f"[ats] atsSampler on: sampler reweight={self._motion_data.reweight} "
                  f"alpha={self.reweight_alpha}",
                  flush=True)

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
        if self._ats_starts is not None and len(env_ids) > 0:
            mids = self.motion_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
            mts = self.motion_times.detach().cpu().numpy().astype(np.int64).reshape(-1)
            for m in np.unique(mids):
                hist = self._ats_starts.get(int(m))
                if hist is not None:
                    np.add.at(hist, np.clip(mts[mids == m], 0, len(hist) - 1), 1)

    def _reweight_motion(self, reset_env_ids):
        super()._reweight_motion(reset_env_ids)
        if not (self._ats_on and self.reweight):
            return
        if not (self.progress_buf_total % self.reweight_intervel == 0 and self.progress_buf_total > 0):
            return
        self._ats_updates += 1
        md = self._motion_data
        for m, rate in md.time_sample_rate.items():
            p = torch.as_tensor(rate, dtype=torch.float64).cpu().numpy()
            ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(len(p)))
            top = (np.argsort(-p)[:5] + 2).tolist()   # rate index i = reference start frame i + 2
            print(f"[ats] step {int(self.progress_buf_total)} motion {m}: sampler reweight={md.reweight} "
                  f"p max {p.max():.4f} min {p.min():.4f} (uniform {1.0 / len(p):.4f}) entropy {ent:.3f} "
                  f"top frames {top}", flush=True)
        if self._ats_updates % 20 == 1 and self._ats_starts is not None:
            for m, hist in self._ats_starts.items():
                n = int(hist.sum())
                if n == 0:
                    continue
                bins = [(a, min(a + 15, len(hist))) for a in range(0, len(hist), 15)]
                txt = " ".join(f"{a}-{b - 1}:{hist[a:b].sum() / n:.2f}" for a, b in bins)
                print(f"[ats] start frames since last histogram (motion {m}, n={n}): {txt}", flush=True)
                hist[:] = 0
