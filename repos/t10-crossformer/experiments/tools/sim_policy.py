"""CrossFormer policy wrapper for SimplerEnv (closed-loop real-to-sim evaluation).

Input/output conversion follows SimplerEnv's own Octo wrapper (simpler_env/policies/octo/octo_model.py), which is
the reference conversion layer for both embodiments:
  observation  camera frame -> 224x224 Lanczos3 (training preprocessing), history window with pad mask
  action       normalized chunk -> dataset statistics (bridge_dataset / fractal20220817_data)
               -> chunk ensembling (uniform average of the overlapping predictions for the current step)
               -> translation * action_scale, euler (roll, pitch, yaw) -> axis-angle * action_scale
  gripper      WidowX: binarize at 0.5 to +1 open / -1 close
               Google Robot: relative command (previous - current, +1 close / -1 open), held ("sticky") for
               15 control steps once it exceeds 0.5
Each of these is a knob in configs/sim_variants.yaml (they are the T10 conversion variables).
"""

from collections import deque

import jax
import numpy as np
from transforms3d.euler import euler2axangle

POLICY_SETUPS = {
    "widowx_bridge": {"dataset": "bridge_dataset", "sticky_repeat": 1},
    "google_robot": {"dataset": "fractal20220817_data", "sticky_repeat": 15},
}


def resize(image, size=224):
    import tensorflow as tf

    image = tf.image.resize(image, size=(size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()


class CrossFormerSimPolicy:
    def __init__(self, model, policy_setup, variant: dict):
        self.model = model
        self.setup = policy_setup
        cfg = POLICY_SETUPS[policy_setup]
        self.stats_name = variant.get("stats", "self")
        stats_ds = cfg["dataset"] if self.stats_name == "self" else self.stats_name
        st = model.dataset_statistics[stats_ds]["action"]
        self.mean, self.std = np.asarray(st["mean"]), np.asarray(st["std"])
        self.mask = np.asarray(st.get("mask", [True] * 7), bool)
        self.task_mode = variant.get("task", "lang")
        self.window = int(variant.get("window", 5))
        self.ensemble = bool(variant.get("ensemble", True))
        self.action_scale = float(variant.get("action_scale", 1.0))
        self.sticky_repeat = int(variant.get("sticky_repeat", cfg["sticky_repeat"]))
        self.rot_mode = variant.get("rotation", "euler_to_axangle")
        self.horizon = 4
        self.history = deque(maxlen=self.window)
        self.chunks = deque(maxlen=self.horizon)
        self.n_seen = 0
        self.task = None
        self.task_description = None
        self.rng = jax.random.PRNGKey(0)  # L1 head is deterministic
        self.log = []  # per-step raw actions for diagnostics

    # --- task -------------------------------------------------------------------------------------------
    def _make_task(self, text):
        if self.task_mode == "lang":
            return self.model.create_tasks(texts=[text])
        if self.task_mode == "none":
            t = self.model.create_tasks(texts=[""])
            t["language_instruction"] = np.zeros_like(np.asarray(t["language_instruction"]))
            t["pad_mask_dict"]["language_instruction"] = np.zeros(1, bool)
            return t
        raise ValueError(f"task mode {self.task_mode} is not available closed-loop (no goal image)")

    def reset(self, task_description):
        self.task = self._make_task(task_description)
        self.task_description = task_description
        self.history.clear()
        self.chunks.clear()
        self.n_seen = 0
        self.sticky_on = False
        self.sticky_count = 0
        self.sticky_action = 0.0
        self.prev_gripper = None
        self.log = []

    # --- step -------------------------------------------------------------------------------------------
    def step(self, image, task_description=None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        self.history.append(resize(image))
        self.n_seen += 1
        frames = list(self.history)
        n_real = len(frames)
        # pad on the left by repeating the first frame, like chunk_act_obs in training
        while len(frames) < self.window:
            frames.insert(0, frames[0])
        mask = np.zeros(self.window, bool)
        mask[self.window - n_real :] = True
        obs = {"image_primary": np.stack(frames)[None], "timestep_pad_mask": mask[None]}
        norm = np.asarray(self.model.sample_actions(obs, self.task, head_name="single_arm", rng=self.rng))[0, :, :7]
        raw = norm.copy()
        raw[:, self.mask] = norm[:, self.mask] * self.std[self.mask] + self.mean[self.mask]

        if self.ensemble:
            self.chunks.append(raw)
            k = len(self.chunks)
            # prediction made i steps ago contributes its i-th chunk element (uniform weights = temperature 0)
            cur = np.mean([c[i] for c, i in zip(self.chunks, range(k - 1, -1, -1))], axis=0)
        else:
            cur = raw[0]

        world_vector = cur[:3] * self.action_scale
        roll, pitch, yaw = np.asarray(cur[3:6], np.float64)
        if self.rot_mode == "euler_to_axangle":
            ax, ang = euler2axangle(roll, pitch, yaw)
            rot = ax * ang * self.action_scale
        elif self.rot_mode == "raw":  # treat the euler deltas as axis-angle (convention mismatch probe)
            rot = np.array([roll, pitch, yaw]) * self.action_scale
        else:
            raise ValueError(self.rot_mode)

        open_g = float(cur[6])
        if self.setup == "google_robot":
            rel = 0.0 if self.prev_gripper is None else self.prev_gripper - open_g
            self.prev_gripper = open_g
            if abs(rel) > 0.5 and not self.sticky_on:
                self.sticky_on, self.sticky_action = True, rel
            if self.sticky_on:
                self.sticky_count += 1
                rel = self.sticky_action
            if self.sticky_count == self.sticky_repeat:
                self.sticky_on, self.sticky_count, self.sticky_action = False, 0, 0.0
            gripper = np.array([rel])
        else:
            gripper = np.array([2.0 * (open_g > 0.5) - 1.0])

        raw_action = {
            "world_vector": np.asarray(cur[:3]),
            "rotation_delta": np.asarray(cur[3:6]),
            "open_gripper": np.asarray(cur[6:7]),
        }
        self.log.append(np.concatenate([cur, world_vector, rot, gripper]))
        action = {
            "world_vector": world_vector,
            "rot_axangle": rot,
            "gripper": gripper,
            "terminate_episode": np.array([0.0]),
        }
        return raw_action, action

    def visualize_epoch(self, predicted_raw_actions, images, save_path):
        # the official evaluator calls this after every episode; we store the raw action log instead of a figure
        np.save(save_path.replace(".png", ".npy"), np.asarray(self.log, np.float32))
