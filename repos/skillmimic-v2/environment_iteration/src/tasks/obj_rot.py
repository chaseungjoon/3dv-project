"""Fix-7 book rotation reward for the own-start task."""

import torch

from env.tasks.skillmimic_parahome import SkillMimicParahome
from tasks.own_start import SkillMimicParahomeIterOwnStart


def object_rotation_factor(sim_quat, ref_quat, weight):
    """Exponential of the squared shortest rotation angle between unit quaternions."""
    cos_half_angle = (sim_quat * ref_quat).sum(dim=-1).abs().clamp(max=1.0)
    angle = 2.0 * torch.acos(cos_half_angle)
    return torch.exp(-weight * angle.square())


class SkillMimicParahomeIterObjRot(SkillMimicParahomeIterOwnStart):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._obj_rot_weight = float(cfg["env"].get("objRotWeight", 0.0))
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

    def _compute_reward(self):
        if self._obj_rot_weight == 0.0:
            return super()._compute_reward()

        # Apply the rotation factor to each raw reward before the inherited buffer-node averaging.
        SkillMimicParahome._compute_reward(self)
        start = 6 + 2 * self._dof_obs_size + 3
        book_rot = slice(start, start + 4)
        self.rew_buf.mul_(object_rotation_factor(
            self._curr_obs[:, book_rot], self._curr_ref_obs[:, book_rot], self._obj_rot_weight
        ))

        # Match SkillMimicParahomeLocalHistRISBuffernode._compute_reward's update order.
        for env_id in torch.nonzero(self.buffer_steps).flatten():
            motion_id = self.motion_ids_total[env_id].item()
            same_motion = self.motion_ids_total == motion_id
            self.rew_buf[env_id] = self.rew_buf[same_motion].mean().item()
        self.buffer_steps = torch.where(self.buffer_steps > 0, self.buffer_steps - 1, self.buffer_steps)
