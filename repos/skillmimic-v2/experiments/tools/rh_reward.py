"""SkillMimic-V2 ParaHome imitation reward, split into its factors (reward-hacking tests, experiments/reward_hacking).

Same formula as skillmimic/env/tasks/skillmimic_parahome.py compute_humanoid_reward (the training reward), written
out in plain torch so every factor can be logged. `reward_parts(...)["rew"]` equals the upstream value; evaluate.py
--log_reward checks that on every step it patches in.

    reward = r_body * r_obj * r_ig
    r_body = r_pos (key body positions, w p=20) * r_rot (root rot + dof, w r=20) * r_smooth (dof velocity change, 0.1)
             (r_pos_vel and r_rot_vel have weight 0 in the configs, so they are always 1)
    r_obj  = r_obj_pos (object position, w op=1)   (object rotation / velocity terms have weight 0)
    r_ig   = interaction graph (every key body point minus the object position, w ig=20)
Each factor is exp(-w * mean squared error). The contact-graph term is commented out upstream (and the ParaHome
reference contact labels are all 0), so it is not part of the reward.
"""
import torch

def layout(dof_obs_size, len_keypos):
    o = 6 + 2 * dof_obs_size
    return dict(dof_pos=slice(6, 6 + dof_obs_size), dof_vel=slice(6 + dof_obs_size, o),
                obj_pos=slice(o, o + 3), obj_rot=slice(o + 3, o + 7), obj_vel=slice(o + 7, o + 10),
                key_pos=slice(o + 10, o + 10 + 3 * len_keypos), size=o + 10 + 3 * len_keypos + 1)


def reward_parts(hoi_ref, hoi_obs, hoi_obs_hist, len_keypos, dof_obs_size, w):
    """hoi_* (N, D) in the hoi layout of build_hoi_observations; w: dict of (N,) tensors (motion_data.reward_weights).
    Returns dict of (N,) tensors: rew and every factor, plus the raw squared errors e_*."""
    L = layout(dof_obs_size, len_keypos)
    n = hoi_obs.shape[0]

    def unpack(h):
        key = torch.cat((h[:, 0:3], h[:, L["key_pos"]]), dim=-1)
        rot = torch.cat((h[:, 3:6], h[:, L["dof_pos"]]), dim=-1)
        obj = h[:, L["obj_pos"]]
        ig = (key.view(n, len_keypos + 1, 3) - obj[:, None, :]).view(n, -1)
        return key, rot, h[:, L["dof_vel"]], obj, h[:, L["obj_rot"]], h[:, L["obj_vel"]], ig

    key, rot, dvel, obj, orot, ovel, ig = unpack(hoi_obs)
    rkey, rrot, rdvel, robj, rorot, rovel, rig_ = unpack(hoi_ref)
    dvel_hist = hoi_obs_hist[:, L["dof_vel"]]

    e_pos = torch.mean((rkey - key) ** 2, dim=-1)
    e_rot = torch.mean((rrot - rot) ** 2, dim=-1)
    e_rv = torch.mean((rdvel - dvel) ** 2, dim=-1)
    e_smooth = torch.mean((dvel - dvel_hist) ** 2 / (((rdvel ** 2) + 1e-12) * 1e12), dim=-1)
    e_obj = torch.mean((robj - obj) ** 2, dim=-1)
    e_orot = torch.mean((rorot - orot) ** 2, dim=-1)
    e_ovel = torch.mean((rovel - ovel) ** 2, dim=-1)
    e_ig = torch.mean((rig_ - ig) ** 2, dim=-1)

    r_pos = torch.exp(-e_pos * w["p"])
    r_rot = torch.exp(-e_rot * w["r"])
    r_pos_vel = torch.exp(-torch.zeros_like(e_pos) * w["pv"])
    r_rot_vel = torch.exp(-e_rv * w["rv"])
    r_smooth = torch.exp(-e_smooth * 0.1)
    r_body = r_pos * r_rot * r_pos_vel * r_rot_vel * r_smooth
    r_obj_pos = torch.exp(-e_obj * w["op"])
    r_obj = r_obj_pos * torch.exp(-e_orot * w["or"]) * torch.exp(-e_ovel * w["opv"]) \
        * torch.exp(-torch.zeros_like(e_pos) * w["orv"])
    r_ig = torch.exp(-e_ig * w["ig"])
    return dict(rew=r_body * r_obj * r_ig, r_body=r_body, r_pos=r_pos, r_rot=r_rot, r_smooth=r_smooth,
                r_obj=r_obj, r_obj_pos=r_obj_pos, r_ig=r_ig, e_pos=e_pos, e_rot=e_rot, e_obj=e_obj, e_ig=e_ig)


PARTS = ["rew", "r_body", "r_pos", "r_rot", "r_smooth", "r_obj", "r_ig"]


def install_logger(task, holder, check_steps=5):
    """Replace the reward function the task calls with reward_parts, so each step's factors land in holder.
    The first `check_steps` calls are compared against the upstream jit function (AssertionError if they differ)."""
    import sys
    owner = next(c for c in type(task).__mro__ if c.__name__ == "SkillMimicParahome")   # defines _compute_reward
    smp = sys.modules[owner.__module__]
    orig = smp.compute_humanoid_reward
    calls = {"n": 0}

    def patched(hoi_ref, hoi_obs, hoi_obs_hist, contact_buf, tar_contact_forces, len_keypos, dof_obs_size, w):
        parts = reward_parts(hoi_ref, hoi_obs, hoi_obs_hist, len_keypos, dof_obs_size, w)
        if calls["n"] < check_steps:
            ref = orig(hoi_ref, hoi_obs, hoi_obs_hist, contact_buf, tar_contact_forces, len_keypos, dof_obs_size, w)
            diff = float((ref - parts["rew"]).abs().max())
            assert diff < 1e-4, f"rh_reward differs from the upstream reward by {diff}"
        calls["n"] += 1
        for k in PARTS:
            holder[k] = parts[k].detach()
        return parts["rew"]

    smp.compute_humanoid_reward = patched
    return calls
