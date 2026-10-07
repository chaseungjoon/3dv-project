"""Per-episode metrics for T09 evaluation (see experiments/docs/PROTOCOL.md, section 4).

EpisodeRecorder only reads environment tensors after each env.step(); it never writes to the
environment, so attaching it cannot change the rollout. All per-frame values are kept for the
whole episode (155 frames x num_envs is small) and reduced at the end.

Frame convention: after the k-th env.step(), task.progress_buf == k, and the simulated state is
compared against reference frame k (task._curr_ref_obs). Recorded frames are 1 .. T-1, where T is
the sequence length; the episode ends when progress reaches T-1 (test_no_reset).
"""
import math

import numpy as np
import torch

from utils import torch_utils

# Fixed before any baseline result was seen (PROTOCOL.md 4.2). Do not tune these on results.
TASK_EPS_M = 0.10            # task success: world object position error must stay below this
FOOT_CONTACT_N = 1.0         # a foot counts as touching the ground above this net contact force
JOINT_LIMIT_MARGIN = 0.035   # rad (~2 deg); a DoF within this of a limit counts as "at limit"
FEET = ['L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe']
TORSO = 'Torso'
LEFT_HAND_IDS = list(range(17, 33))    # L_Wrist + 15 left finger bodies (same as compute_cg_reward)
RIGHT_HAND_IDS = list(range(36, 52))   # R_Wrist + 15 right finger bodies
HANDS = ('L', 'R')


def _quat_angle(q_ref, q):
    diff = torch_utils.quat_mul_norm(torch_utils.quat_inverse(q_ref), q)
    angle, _ = torch_utils.quat_to_angle_axis(diff)
    return angle


def _tilt(q):
    """Angle (rad) between the body's local +z axis and world +z."""
    up = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    up[..., 2] = 1.0
    up_w = torch_utils.quat_rotate(q.reshape(-1, 4), up.reshape(-1, 3)).reshape(up.shape)
    return torch.acos(torch.clamp(up_w[..., 2], -1.0, 1.0))


class EpisodeRecorder:
    def __init__(self, task):
        self.task = task
        self.n = task.num_envs
        self.dt = float(task.dt)
        self.feet_ids = task._build_body_ids_tensor(FEET)
        self.torso_id = int(task._build_body_ids_tensor([TORSO])[0])
        self.key_ids = task._key_body_ids
        # DoF layout: body b (b >= 1) owns DoFs (b-1)*3 .. (b-1)*3+2
        hand_dof = torch.zeros(task.num_dof, dtype=torch.bool, device=task.device)
        for b in task._finger_body_ids.tolist():
            hand_dof[(b - 1) * 3:(b - 1) * 3 + 3] = True
        self.hand_dof = hand_dof
        self.lo = task.dof_limits_lower
        self.hi = task.dof_limits_upper
        self.seq_len = task.max_episode_length[task.data_id].clone()  # [N]
        self.frames = []

    @torch.no_grad()
    def step(self, info, reward):
        t = self.task
        f = {}
        f['frame'] = t.progress_buf.clone()
        gate = t.progress_buf > 1 + t.start_times

        tc = info['termination_causes']
        kc = info['kinematic_causes']
        wd = info['wrist_diag']
        f['fall'] = tc['fall'].clone()
        f['kinematic'] = tc['kinematic'].clone()
        f['contact'] = tc['contact'].clone()
        f['body'] = kc['body'] & gate
        f['object'] = kc['object'] & gate
        f['wrist_hand'] = wd['reset'] & gate.unsqueeze(-1)          # [N, 2] (L, R)
        f['any_fail'] = f['fall'] | f['kinematic'] | f['contact']
        f['contact_mismatch'] = t.contact_reset.clone()             # consecutive frames, [N, 2]
        f['official_terminate'] = info['terminate'].clone().bool()

        # wrist tracking (same quantities the reset condition uses)
        f['wrist_pos_err'] = wd['pos_diff'].clone()
        f['wrist_rot_err'] = wd['rot_diff'].clone()
        f['wrist_stage'] = wd['stage'].clone()
        f['hand_active'] = wd['has_contact'].clone()

        # object, world frame
        ref = t._curr_ref_obs
        obj_pos = t._target_states[:, 0, 0:3]
        obj_rot = t._target_states[:, 0, 3:7]
        ref_obj_pos = t.extract_data_component('obj_pos', obs=ref)
        ref_obj_rot = t.extract_data_component('obj_rot', obs=ref)
        f['obj_pos_err'] = (obj_pos - ref_obj_pos).norm(dim=-1)
        f['obj_rot_err'] = _quat_angle(ref_obj_rot, obj_rot)
        # the official printout's numbers (heading-local frame, see CODE_NOTES.md)
        f['obj_pos_err_local'] = info['obj_pos_error'].clone()
        f['obj_rot_err_local'] = info['obj_rot_error'].clone()

        # whole-body stability
        body_pos = t._rigid_body_pos
        body_rot = t._rigid_body_rot
        f['pelvis_h'] = body_pos[:, 0, 2].clone()
        ref_body_pos = t.extract_data_component('body_pos', obs=ref).view(self.n, -1, 3)
        ref_body_rot = t.extract_data_component('body_rot', obs=ref).view(self.n, -1, 4)
        f['key_body_err'] = (body_pos[:, self.key_ids] - ref_body_pos[:, self.key_ids]).norm(dim=-1).mean(dim=-1)
        f['tilt'] = _tilt(body_rot[:, self.torso_id])
        f['tilt_ref'] = _tilt(ref_body_rot[:, self.torso_id])
        feet_force = t._contact_forces[:, self.feet_ids].norm(dim=-1)                 # [N, 4]
        feet_vxy = t._rigid_body_vel[:, self.feet_ids, 0:2].norm(dim=-1)               # [N, 4]
        f['foot_contact'] = feet_force > FOOT_CONTACT_N
        f['foot_slip_speed'] = feet_vxy * f['foot_contact'].float()
        q = t._dof_pos
        at_limit = ((q - self.lo) < JOINT_LIMIT_MARGIN) | ((self.hi - q) < JOINT_LIMIT_MARGIN)
        f['jl_body'] = at_limit[:, ~self.hand_dof].float().mean(dim=-1)
        f['jl_hand'] = at_limit[:, self.hand_dof].float().mean(dim=-1)

        # hand-object contact (same definition as compute_cg_reward)
        cur = t._curr_obs
        hum = t.extract_data_component('contact_human', obs=cur)
        ref_hum = t.extract_data_component('contact_human', obs=ref)
        obj_c = t.extract_data_component('contact_obj', obs=cur) > 0.1
        ref_obj_c = t.extract_data_component('contact_obj', obs=ref) > 0.1
        sim_hand, ref_hand = [], []
        for ids in (LEFT_HAND_IDS, RIGHT_HAND_IDS):
            sim_hand.append(((hum[:, ids] * obj_c) > 0.1).any(dim=-1))
            ref_hand.append(((ref_hum[:, ids] * ref_obj_c) > 0.1).any(dim=-1))
        f['hand_contact_sim'] = torch.stack(sim_hand, dim=-1)
        f['hand_contact_ref'] = torch.stack(ref_hand, dim=-1)

        f['reward'] = reward.clone().float().view(-1)
        self.frames.append(f)

    # ------------------------------------------------------------------ reduction
    def series(self):
        """All per-frame values as numpy arrays shaped [frames, N, ...]."""
        keys = self.frames[0].keys()
        return {k: torch.stack([fr[k] for fr in self.frames]).cpu().numpy() for k in keys}

    def summarize(self):
        s = self.series()
        out = []
        for e in range(self.n):
            out.append(_summarize_env({k: v[:, e] for k, v in s.items()}, int(self.seq_len[e]), self.dt))
        return out, s


def _first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1


def _mean(x, mask=None):
    if mask is not None:
        x = x[mask]
    return float(np.mean(x)) if np.size(x) else float('nan')


def _summarize_env(s, seq_len, dt):
    frames = s['frame']
    n = len(frames)
    last = n - 1
    r = {'seq_len': seq_len, 'frames_recorded': n, 'last_frame': int(frames[-1])}

    # ---------------- official success (any termination over the whole sequence) ----------------
    i_fail = _first(s['any_fail'])
    r['official_success'] = int(i_fail < 0)
    r['official_success_env_flag'] = int(not s['official_terminate'][last])
    r['first_fail_frame'] = int(frames[i_fail]) if i_fail >= 0 else -1
    # progress = fraction of the sequence completed before the first failure
    r['progress'] = 1.0 if i_fail < 0 else float(frames[i_fail] - 1) / float(seq_len - 1)
    causes = {
        'fall': s['fall'], 'body': s['body'], 'object': s['object'],
        'wrist_L': s['wrist_hand'][:, 0], 'wrist_R': s['wrist_hand'][:, 1], 'contact': s['contact'],
    }
    for name, m in causes.items():
        r[f'first_cause_{name}'] = int(i_fail >= 0 and bool(m[i_fail]))
        r[f'frames_{name}'] = int(np.sum(m))
    # kinematic flag with no recorded sub-cause would point at a logging bug
    r['first_cause_unexplained'] = int(i_fail >= 0 and not any(bool(m[i_fail]) for m in causes.values()))

    # ---------------- object ----------------
    pe, re_ = s['obj_pos_err'], s['obj_rot_err']
    r['obj_pos_err_mean'] = _mean(pe)
    r['obj_pos_err_max'] = float(np.max(pe))
    r['obj_pos_err_final'] = float(pe[last])
    r['obj_rot_err_mean'] = _mean(re_)
    r['obj_rot_err_max'] = float(np.max(re_))
    r['obj_pos_err_local_mean'] = _mean(s['obj_pos_err_local'])
    r['obj_rot_err_local_mean'] = _mean(s['obj_rot_err_local'])

    # ---------------- task success (wrist-agnostic) ----------------
    fell = s['fall'].any()
    r['fell'] = int(fell)
    r['task_success'] = int((r['obj_pos_err_max'] <= TASK_EPS_M) and not fell)
    i_task = _first((pe > TASK_EPS_M) | s['fall'])
    r['task_progress'] = 1.0 if i_task < 0 else float(frames[i_task] - 1) / float(seq_len - 1)

    # ---------------- wrist ----------------
    active = []
    for h, name in enumerate(HANDS):
        pos, rot, stage = s['wrist_pos_err'][:, h], s['wrist_rot_err'][:, h], s['wrist_stage'][:, h]
        is_active = bool(s['hand_active'][0, h])
        r[f'wrist_{name}_active'] = int(is_active)
        r[f'wrist_{name}_pos_err_mean'] = _mean(pos)
        r[f'wrist_{name}_rot_err_mean'] = _mean(rot)
        r[f'wrist_{name}_pos_err_window'] = _mean(pos, stage > 0)
        r[f'wrist_{name}_rot_err_window'] = _mean(rot, stage > 0)
        r[f'wrist_{name}_pos_err_stage2_max'] = float(np.max(pos[stage == 2])) if np.any(stage == 2) else float('nan')
        r[f'wrist_{name}_rot_err_stage2_max'] = float(np.max(rot[stage == 2])) if np.any(stage == 2) else float('nan')
        r[f'wrist_{name}_reset_frames'] = int(np.sum(s['wrist_hand'][:, h]))
        if is_active:
            active.append(name)
    for key in ('pos_err_mean', 'rot_err_mean', 'pos_err_window', 'rot_err_window', 'reset_frames'):
        vals = [r[f'wrist_{h}_{key}'] for h in active]
        r[f'wrist_{key}'] = float(np.mean(vals)) if vals else float('nan')

    # ---------------- stability ----------------
    r['pelvis_h_min'] = float(np.min(s['pelvis_h']))
    r['key_body_err_mean'] = _mean(s['key_body_err'])
    slip = s['foot_slip_speed']                       # [frames, 4], zero when not in contact
    contact = s['foot_contact']
    r['foot_slip_dist'] = float(np.sum(slip) * dt)    # metres, summed over the four foot bodies
    r['foot_slip_speed_contact'] = float(np.sum(slip) / max(1, np.sum(contact)))
    r['joint_limit_body'] = _mean(s['jl_body'])
    r['joint_limit_hand'] = _mean(s['jl_hand'])
    tilt, tilt_ref = np.degrees(s['tilt']), np.degrees(s['tilt_ref'])
    r['torso_tilt_max_deg'] = float(np.max(tilt))
    r['torso_tilt_excess_mean_deg'] = _mean(np.abs(tilt - tilt_ref))
    r['torso_tilt_excess_max_deg'] = float(np.max(np.abs(tilt - tilt_ref)))

    # ---------------- grasp contact ----------------
    ref_c, sim_c = s['hand_contact_ref'], s['hand_contact_sim']
    hands = [h for h, name in enumerate(HANDS) if r[f'wrist_{name}_active']]
    ref_n = int(sum(np.sum(ref_c[:, h]) for h in hands))
    both = int(sum(np.sum(ref_c[:, h] & sim_c[:, h]) for h in hands))
    r['contact_ref_frames'] = ref_n
    r['contact_recall'] = float(both / ref_n) if ref_n else float('nan')

    r['reward_sum'] = float(np.sum(s['reward']))
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
