"""CPU checks for the fix-8 state sync (src/tasks/state_sync.py): reference row / frame per state index, the motion-file
tail row, the legacy _curr_ref_obs after the step, the own-start capture frame and error, delegation when off."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC.parents[1] / "skillmimic")]

import isaacgym  # noqa: F401, E402 (must be imported before torch)
import torch  # noqa: E402

import tasks.state_sync as SS  # noqa: E402
from tasks.state_sync import OBJ_POS, SkillMimicParahomeIterOwnStart, SkillMimicParahomeIterStateSync  # noqa: E402

W = OBJ_POS + 3 + 4      # row width used by the fake data (book position at OBJ_POS)
CLIP = 20                # motion frames


class FakeMotion:
    def __init__(self, sframe, elen):
        n = len(sframe)
        rows = torch.arange(CLIP, dtype=torch.float).unsqueeze(-1).repeat(1, W)   # row f == f everywhere
        self.hoi_data_dict = {0: {"hoi_data": rows}}
        self.envid2sframe = torch.tensor(sframe)
        self.envid2episode_lengths = torch.tensor(elen)
        self.envid2motid = torch.zeros(n, dtype=torch.long)
        self.motion_lengths = torch.tensor([CLIP])


def make_task(sframe, elen, progress, maxlen=6):
    t = SkillMimicParahomeIterStateSync.__new__(SkillMimicParahomeIterStateSync)
    n = len(sframe)
    t._sync, t._sync_ready, t.isTest = True, True, False
    t.num_envs, t.device, t._dof_obs_size = n, "cpu", 180
    t._motion_data = md = FakeMotion(sframe, elen)
    t.progress_buf = torch.tensor(progress)
    t.max_episode_length = maxlen
    t.hoi_data_batch = torch.zeros(n, maxlen, W)
    for e in range(n):                       # upstream slice + zero padding
        k = elen[e]
        t.hoi_data_batch[e, :k] = md.hoi_data_dict[0]["hoi_data"][sframe[e]:sframe[e] + k]
    t._curr_ref_obs = torch.full((n, W), -7.0)
    t._curr_obs = torch.zeros(n, W)
    t._hist_obs = torch.zeros(n, W)
    t._legacy_ref_obs = torch.zeros(n, W)
    t._sync_ref_obs = torch.zeros(n, W)
    t._sync_ref_frame = torch.zeros(n, dtype=torch.long)
    t._ref_tail = torch.zeros(n, W)
    t._dof_vel = torch.zeros(n, 180)
    nb = 3
    t._rigid_body_pos = torch.zeros(n, nb, 3)
    t._rigid_body_rot = torch.zeros(n, nb, 4)
    t._rigid_body_vel = torch.zeros(n, nb, 3)
    t._rigid_body_ang_vel = torch.zeros(n, nb, 3)
    t._key_body_ids = torch.tensor([1, 2])
    t._dof_pos = torch.zeros(n, 180)
    t._target_states = torch.zeros(n, 13)
    t._local_root_obs, t._root_height_obs = False, True
    t._sync_stats_reset()
    t._sync_after_reset(torch.arange(n))
    return t


class StateSyncTest(unittest.TestCase):
    def test_reference_row_frame_and_tail(self):
        # env0: inside the batch; env1: last state of a 60-cap style episode (tail = sframe+E);
        # env2: last state at the clip end (tail held at the clip's last frame)
        t = make_task(sframe=[2, 3, 16], elen=[6, 4, 4], progress=[1, 3, 3])
        seen = {}

        def build(*args):
            seen["idx"] = args[-1].clone()
            return torch.zeros(t.num_envs, W)
        with patch.object(SS, "build_hoi_observations", build):
            t._sync_snapshot()
        self.assertEqual(seen["idx"].tolist(), [2, 4, 4])               # state index = progress + 1
        self.assertEqual(t._sync_ref_frame.tolist(), [4, 7, CLIP - 1])
        self.assertEqual(t._sync_ref_obs[:, 0].tolist(), [4.0, 7.0, CLIP - 1.0])
        torch.testing.assert_close(t._curr_ref_obs, t._sync_ref_obs)
        torch.testing.assert_close(t._legacy_ref_obs, torch.full((3, W), -7.0))
        self.assertEqual(t._sync_ref_check(), 0.0)
        self.assertEqual(int(t._sync_stats["tail"]), 2)

    def test_reward_sees_sync_reference_and_step_restores_legacy(self):
        t = make_task(sframe=[2, 5], elen=[6, 6], progress=[0, 2])
        seen = {}

        def parent_reward(self_):
            seen["ref"] = self_._curr_ref_obs[:, 0].clone()

        def parent_step(self_):
            self_._curr_ref_obs[:] = self_.hoi_data_batch[torch.arange(2), self_.progress_buf][:, :]  # legacy write
            self_._compute_reward()
            self_.progress_buf += 1
        with patch.object(SS, "build_hoi_observations", lambda *a: torch.zeros(2, W)), \
                patch.object(SkillMimicParahomeIterOwnStart, "_compute_reward", parent_reward), \
                patch.object(SkillMimicParahomeIterOwnStart, "post_physics_step", parent_step):
            t.post_physics_step()
        self.assertEqual(seen["ref"].tolist(), [3.0, 8.0])           # reward: frame sframe + k + 1
        self.assertEqual(t._curr_ref_obs[:, 0].tolist(), [2.0, 7.0])  # after the step: legacy sframe + k

    def test_capture_uses_snapshot_frame_and_error(self):
        t = make_task(sframe=[2, 2], elen=[6, 6], progress=[3, 3])
        t._own_flag = torch.zeros(2, dtype=torch.bool)
        t._own_max_start, t._own_frames, t._own_obj_dist = 60, [0, 100], 0.5
        t._own_per_step, t._own_cap = 8, 4
        t._terminate_buf = torch.zeros(2, dtype=torch.long)
        t._sync_ref_frame = torch.tensor([5, 5])
        t._sync_ref_obs = t._motion_data.hoi_data_dict[0]["hoi_data"][[5, 5]].clone()
        t._curr_obs = t._sync_ref_obs.clone()
        t._curr_obs[0, OBJ_POS] += 0.3                                # env0 book 0.3 m off, env1 on track
        t._target_states = torch.zeros(2, 13)
        t._target_states[:, 0:3] = t._curr_obs[:, OBJ_POS:OBJ_POS + 3]
        t._humanoid_root_states = torch.zeros(2, 13)
        t._dof_pos, t._dof_vel = torch.zeros(2, 4), torch.zeros(2, 4)
        t._hist_obs_batch = torch.zeros(2, 3, 5)
        t._ob_root, t._ob_obj = torch.zeros(4, 13), torch.zeros(4, 13)
        t._ob_dof_pos, t._ob_dof_vel = torch.zeros(4, 4), torch.zeros(4, 4)
        t._ob_hist = torch.zeros(4, 3, 5)
        t._ob_frame, t._ob_motion = torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)
        t._ob_err, t._ob_n, t._own_stats = torch.zeros(4), 0, {"captures": 0}
        t._own_capture()
        self.assertEqual(t._ob_n, 2)
        self.assertEqual(t._ob_frame[:2].tolist(), [5, 5])
        torch.testing.assert_close(t._ob_err[:2], torch.tensor([0.3, 0.0]))
        self.assertLess(float(t._sync_stats["cap_check"]), 1e-6)

    def test_off_delegates_to_parent(self):
        t = SkillMimicParahomeIterStateSync.__new__(SkillMimicParahomeIterStateSync)
        t._sync = False
        with patch.object(SkillMimicParahomeIterOwnStart, "_compute_hoi_observations") as h, \
                patch.object(SkillMimicParahomeIterOwnStart, "_own_capture") as c, \
                patch.object(SkillMimicParahomeIterOwnStart, "_compute_reward") as r:
            t._compute_hoi_observations()
            t._own_capture()
            t._compute_reward()
        h.assert_called_once_with(None)
        c.assert_called_once_with()
        r.assert_called_once_with()

    def test_pre_refresh_build_is_skipped_when_on(self):
        t = SkillMimicParahomeIterStateSync.__new__(SkillMimicParahomeIterStateSync)
        t._sync = True
        with patch.object(SkillMimicParahomeIterOwnStart, "_compute_hoi_observations") as h:
            t._compute_hoi_observations()
            t._compute_hoi_observations(torch.tensor([0]))
        h.assert_called_once()                    # only the explicit env_ids call reaches the parent


if __name__ == "__main__":
    unittest.main()
