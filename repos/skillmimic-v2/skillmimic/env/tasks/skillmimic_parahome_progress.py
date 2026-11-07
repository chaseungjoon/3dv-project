"""Opt-in book progress experiment. Original fix task remains unchanged."""
import json
import math
import os

import torch
from utils.object_progress import normalized_progress
from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRISBuffernodePhase


class SkillMimicParahomeProgress(SkillMimicParahomeLocalHistRISBuffernodePhase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._motion_data.num_motions != 1:
            raise ValueError('Progress experiment requires one book clip')
        ref = self._motion_data.hoi_data_dict[0]['obj_pos']
        if not torch.isfinite(ref).all():
            raise ValueError('Non-finite reference object positions')
        self._progress_goal = ref[-1].clone()
        self._progress_length = (ref[-1] - ref[0]).norm().item()
        if self._progress_length < 0.05:
            raise ValueError('Endpoint distance too small; do not reuse book reward for cup')
        self._progress_weight = float(os.environ.get('SMV2_PROGRESS_WEIGHT', '0'))
        self._progress_clip = float(os.environ.get('SMV2_PROGRESS_CLIP', '0.05'))
        if not math.isfinite(self._progress_weight) or self._progress_weight < 0:
            raise ValueError('Invalid progress weight')
        if not math.isfinite(self._progress_clip) or self._progress_clip <= 0:
            raise ValueError('Invalid progress clip')
        self._progress_log = os.environ.get('SMV2_PROGRESS_LOG')
        self._progress_calls = 0
        self._progress_sums = torch.zeros(5, device=self.device)
        self._progress_before = None
        if self._progress_log:
            with open(self._progress_log, 'a') as f:
                f.write(json.dumps({'event': 'goal', 'position_m': ref[-1].tolist(),
                                   'normalization_m': self._progress_length,
                                   'weight': self._progress_weight, 'clip': self._progress_clip}) + '\n')

    def pre_physics_step(self, actions):
        # Root state is authoritative immediately after indexed resets; avoid stale rigid-body state.
        self._progress_before = self._target_states[:, :3].clone()
        super().pre_physics_step(actions)

    def _compute_reward(self):
        masked = self.buffer_steps > 0  # parent decrements this counter
        super()._compute_reward()       # preserve original fix and buffer replacement
        if not hasattr(self, '_progress_sums') or self._progress_before is None:
            return
        base = self.rew_buf.clone()
        after = self._target_states[:, :3]
        raw = normalized_progress(
            (self._progress_before - self._progress_goal).norm(dim=-1),
            (after - self._progress_goal).norm(dim=-1),
            self._progress_length, self._progress_clip)
        raw = torch.where(masked, torch.zeros_like(raw), raw)
        # Exactly preserve original buffer values when weight=0 (no unnecessary addition).
        added = raw * self._progress_weight
        if self._progress_weight != 0:
            self.rew_buf.add_(added)
        if not torch.isfinite(self.rew_buf).all():
            raise FloatingPointError('Non-finite reward')
        # Terminal transition is counted before reset; no cross-episode cached distance.
        self._progress_sums += torch.stack([base.mean(), raw.mean(), raw.abs().mean(),
                                           added.mean(), self.rew_buf.mean()]).detach()
        self._progress_calls += 1
        if self._progress_log and self._progress_calls % 32 == 0:
            values = (self._progress_sums / 32).tolist()
            row = dict(zip(['base_mean', 'progress_mean', 'progress_abs_mean',
                            'added_mean', 'total_mean'], values))
            row.update(event='reward', control_steps=self._progress_calls)
            with open(self._progress_log, 'a') as f:
                f.write(json.dumps(row) + '\n')
            self._progress_sums.zero_()
