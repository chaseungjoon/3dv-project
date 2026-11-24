"""CPU checks for the fix-7 reward and its buffer-node ordering."""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC.parents[1] / "skillmimic")]

import isaacgym  # noqa: F401, E402 (must be imported before torch)
import torch  # noqa: E402

from tasks.obj_rot import (  # noqa: E402
    SkillMimicParahome,
    SkillMimicParahomeIterObjRot,
    SkillMimicParahomeIterOwnStart,
    object_rotation_factor,
)


class ObjectRotationRewardTest(unittest.TestCase):
    def test_shortest_angle_and_quaternion_sign(self):
        angles = torch.tensor([0.0, math.pi / 9, math.pi / 2, math.pi], dtype=torch.float64)
        half = angles / 2
        zeros = torch.zeros_like(half)
        sim = torch.stack([zeros, zeros, half.sin(), half.cos()], dim=-1)
        ref = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=sim.dtype).expand_as(sim)

        for weight in (0.5, 1.0):
            expected = torch.exp(-weight * angles.square())
            for sim_sign, ref_sign in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                actual = object_rotation_factor(sim_sign * sim, ref_sign * ref, weight)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        # Unit quaternion dots can round above 1.
        clamped = object_rotation_factor(ref * (1 + 1e-7), ref, 1.0)
        torch.testing.assert_close(clamped, torch.ones_like(clamped))

    def test_rotation_precedes_buffer_averaging(self):
        raw_reward = torch.tensor([0.8, 0.4, 0.6, 0.2])
        task = SkillMimicParahomeIterObjRot.__new__(SkillMimicParahomeIterObjRot)
        task._obj_rot_weight = 1.0
        task._dof_obs_size = 3
        start = 6 + 2 * task._dof_obs_size + 3
        task._curr_ref_obs = torch.zeros(4, start + 4)
        task._curr_ref_obs[:, start + 3] = 1
        task._curr_obs = task._curr_ref_obs.clone()
        task._curr_obs[[0, 3], start + 2:start + 4] = math.sqrt(0.5)
        task.rew_buf = torch.empty_like(raw_reward)
        task.buffer_steps = torch.tensor([1, 0, 2, 0])
        task.motion_ids_total = torch.tensor([0, 0, 1, 1])

        with patch.object(SkillMimicParahome, "_compute_reward", lambda self: self.rew_buf.copy_(raw_reward)):
            task._compute_reward()

        factor = math.exp(-(math.pi / 2) ** 2)
        torch.testing.assert_close(task.rew_buf, torch.tensor([
            (0.8 * factor + 0.4) / 2, 0.4, (0.6 + 0.2 * factor) / 2, 0.2 * factor
        ]))
        torch.testing.assert_close(task.buffer_steps, torch.tensor([0, 0, 1, 0]))

    def test_absent_weight_delegates_to_parent(self):
        with patch.object(SkillMimicParahomeIterOwnStart, "__init__", return_value=None):
            task = SkillMimicParahomeIterObjRot({"env": {}}, None, None, "cpu", 0, True)
        with patch.object(SkillMimicParahomeIterOwnStart, "_compute_reward") as parent_reward:
            task._compute_reward()
        parent_reward.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
