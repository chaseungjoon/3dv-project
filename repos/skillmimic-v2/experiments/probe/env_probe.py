"""Probe the real training environment without a policy (diagnostics in project root SKILLMIMIC_NOTES.md 4).

    .venv/bin/python experiments/probe/env_probe.py --clip drink_cup [--method ours|fix] [--envs 4] [--only MODE]

Builds the task exactly as training does (methods.py flags; SMV2_CFG_ENV swaps the env yaml), then by default:
  1. replay: teleports the humanoid along the reference (root + dof pos/vel every step) and lets the object
     simulate. Perfect tracking with this hand model: does the object follow the reference?
  2. static: humanoid parked far away, object at its frame-0 reference pose. Does the object stay put?
  3. history: history-encoder input built the way the task builds it during rollout vs the way it is filled
     at reset (and trained), on the same reference frames; effect on the 3-d embedding.
--only MODE runs one extra test instead:
  hits      replay; per env, the first frame the object moves and which bodies are near / in contact
  gap       distance from the reference hand bodies to the object mesh (does the demonstration touch it)
  phase     how much clip phase the history embedding carries (buggy vs fixed history layout)
  gripN     start in the reference state of frame N with the object in hand, root pinned, PD to the reference
            pose (+ finger closing variants, or per-env object shape values via PROBE_SHAPE="friction:1,2,4")
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "tools"))
import methods as M  # noqa: E402


def build(clip, method, envs):
    os.chdir(REPO)
    sys.argv = ["skillmimic/run.py", "--test"] + M.base_args(clip, method) + [
        "--num_envs", str(envs), "--episode_length", "300", "--state_init", "2", "--headless",
        "--checkpoint", os.environ.get("PROBE_CKPT", "experiments/runs/drink_cup_ours_n2048_s0/nn/drink_cup_ours_n2048_s0_00003000.pth")]
    sys.path.insert(0, os.path.join(REPO, "skillmimic"))
    from isaacgym import gymapi  # noqa: F401
    from rl_games.torch_runner import Runner
    import run as R
    captured = {}
    Runner.run = lambda self, args: captured.setdefault("runner", self)
    R.main()
    player = captured["runner"].create_player()
    return player.env.task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="drink_cup")
    ap.add_argument("--method", default="ours")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--only", default="", help="hits | gap | phase | grip<frame> (see module doc)")
    a = ap.parse_args()
    task = build(a.clip, a.method, a.envs)
    if a.only == "hits":
        return who_hits(task, task._motion_data.hoi_data_dict[0])
    if a.only.startswith("grip"):
        fr = int(a.only[4:] or 60)
        return grip(task, task._motion_data.hoi_data_dict[0], frame=fr)
    if a.only == "phase":
        return he_phase(task, task._motion_data.hoi_data_dict[0])
    if a.only == "gap":
        return hand_gap(task, task._motion_data.hoi_data_dict[0], M.CLIPS[a.clip]["obj"])
    import numpy as np
    import torch
    from isaacgym import gymtorch
    gym, sim = task.gym, task.sim
    md = task._motion_data.hoi_data_dict[0]
    T = min([int(task._motion_data.motion_lengths[0])] + [md[k].shape[0] for k in ('root_pos', 'root_rot', 'root_pos_vel', 'root_rot_vel', 'dof_pos', 'dof_pos_vel', 'obj_pos', 'obj_rot', 'key_body_pos')])
    n = task.num_envs
    hum_ids = task._humanoid_actor_ids
    tar_ids = task._tar_actor_ids
    from scipy.spatial.transform import Rotation as Rot

    def set_humanoid(t, park=False):
        rs = task._humanoid_root_states
        rs[:, 0:3] = md['root_pos'][t]
        if park:
            rs[:, 0] += 5.0
        rs[:, 3:7] = md['root_rot'][t]
        rs[:, 7:10] = md['root_pos_vel'][t]
        rs[:, 10:13] = md['root_rot_vel'][t]
        task._dof_pos[:] = md['dof_pos'][t]
        task._dof_vel[:] = md['dof_pos_vel'][t]
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states),
                                                gymtorch.unwrap_tensor(hum_ids), n)
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._dof_state),
                                         gymtorch.unwrap_tensor(hum_ids), n)

    def set_object(t):
        ts = task._target_states
        ts[:, 0:3] = md['obj_pos'][t]
        ts[:, 3:7] = md['obj_rot'][t]
        ts[:, 7:13] = 0
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states),
                                                gymtorch.unwrap_tensor(tar_ids), n)

    def step():
        task._dof_state  # noqa
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        task._refresh_sim_tensors()

    ref_obj = md['obj_pos'][:T].cpu().numpy()
    lift_start = int(np.argmax(ref_obj[:, 2] - ref_obj[0, 2] > 0.02))
    mass = sum(p.mass for p in gym.get_actor_rigid_body_properties(task.envs[0], task._target_handles[0]))
    print(f"\n[probe] clip {a.clip}: {T} frames, reference lift starts at frame {lift_start}, object mass {mass:.3f} kg")
    print(f"[probe] humanoid bodies in sim: {task.num_bodies}, reference bodies: {md['body_pos'].shape[1]}")

    # ---- 1. replay ----
    set_humanoid(0); set_object(0); step()
    obj, kerr, hand_f = [], [], []
    key = task._key_body_ids
    for t in range(1, T):
        set_humanoid(t)
        step()
        obj.append(task._target_states[:, 0:3].cpu().numpy().copy())
        kp = task._rigid_body_pos[:, key].cpu().numpy()
        kerr.append(np.linalg.norm(kp - md['key_body_pos'][t].view(-1, 3).cpu().numpy()[None], axis=-1).mean())
        hand_f.append(task._contact_forces[:, task._contact_body_ids].norm(dim=-1).sum(-1).mean().item())
    obj = np.stack(obj)
    err = np.linalg.norm(obj - ref_obj[1:T][:, None], axis=-1).mean(-1)
    lift = obj[:, :, 2].mean(-1) - obj[0, :, 2].mean()
    ref_lift = ref_obj[1:T, 2] - ref_obj[1, 2]
    print(f"[replay] teleport check: key-body error vs reference {np.mean(kerr)*100:.2f} cm (should be ~0)")
    print("[replay] frame | obj err cm | obj lift cm (ref) | fingertip contact force N")
    for t in sorted(set(list(range(0, T - 1, 10)) + [lift_start - 1])):
        print(f"           {t+1:4d} | {err[t]*100:9.1f} | {lift[t]*100:6.1f} ({ref_lift[t]*100:6.1f}) | {hand_f[t]:8.1f}")
    frac = float((err < 0.10).mean())
    print(f"[replay] frames with object within 10 cm: {frac*100:.0f}%, max lift {lift.max()*100:.1f} cm "
          f"(reference {ref_lift.max()*100:.1f} cm)")

    # ---- 2. static: humanoid parked 5 m away ----
    set_humanoid(0, park=True); set_object(0); step()
    z0 = None
    traj = []
    for t in range(60):
        set_humanoid(0, park=True)
        step()
        traj.append(task._target_states[:, 0:3].cpu().numpy().copy())
    traj = np.stack(traj)
    d = np.linalg.norm(traj - ref_obj[0][None, None], axis=-1)
    print(f"[static] object vs its frame-0 reference after 1/10/60 steps (mean over envs): "
          f"{d[0].mean()*100:.1f} / {d[9].mean()*100:.1f} / {d[-1].mean()*100:.1f} cm; "
          f"z offset at rest {(traj[-1, :, 2].mean() - ref_obj[0, 2])*100:+.1f} cm")

    # ---- 3. history encoder input ----
    if hasattr(task, "hist_encoder"):
        from env.tasks.skillmimic_parahome_localhist import get_humanoid_obs, get_obj_obs
        ref_rows, sim_rows = [], []
        set_humanoid(0); set_object(0); step()
        for t in range(T):
            set_humanoid(t)
            if t > 0:
                step()
            ho = task._compute_humanoid_obs(None)
            oo = task._compute_obj_obs(None)
            sim_rows.append(torch.cat([ho[0, :211], task._dof_pos[0], oo[0, :3]]))
        ref_h = get_humanoid_obs(md['root_pos'], md['root_rot_3d'], md['body_pos'])
        ref_o = get_obj_obs(md['root_pos'], md['root_rot_3d'], md['obj_pos'])
        ref_all = torch.cat([ref_h, md['dof_pos'].view(-1, 180), ref_o], -1)[:T]
        sim_all = torch.stack(sim_rows)
        L = task.history_length
        seg = {"root h + 60 body pos (0:181)": slice(0, 181), "dims 181:211": slice(181, 211),
               "dof pos (211:391)": slice(211, 391), "object (391:394)": slice(391, 394)}
        print("[history] per-segment mean |sim layout - reference layout| on identical states:")
        for k, s in seg.items():
            print(f"           {k:30s} {(sim_all[:, s] - ref_all[:, s]).abs().mean().item():.4f}"
                  f"  (reference magnitude {ref_all[:, s].abs().mean().item():.4f})")
        with torch.no_grad():
            wins_ref = torch.stack([ref_all[t - L:t] for t in range(L, T)])
            wins_sim = torch.stack([sim_all[t - L:t] for t in range(L, T)])
            e_ref = task.hist_encoder(wins_ref)
            e_sim = task.hist_encoder(wins_sim)
            fixed = wins_sim.clone(); fixed[..., 181:211] = wins_ref[..., 181:211]
            e_fix = task.hist_encoder(fixed)
        spread = e_ref.std(0).mean().item()
        print(f"[history] embedding spread over the clip (reference windows): {spread:.4f}")
        print(f"[history] |emb(sim-layout windows) - emb(reference windows)| = {(e_sim - e_ref).abs().mean().item():.4f}"
              f"   with dims 181:211 taken from reference: {(e_fix - e_ref).abs().mean().item():.4f}")
        print(f"[history] reference emb first/last: {e_ref[0].tolist()} / {e_ref[-1].tolist()}")
        print(f"[history] sim-layout emb first/last: {e_sim[0].tolist()} / {e_sim[-1].tolist()}")



def who_hits(task, md, frames=40):
    """Replay; per env: first frame the object moves > 3 cm in xy, and what is near / in contact then."""
    import numpy as np
    from isaacgym import gymtorch
    gym, sim, n = task.gym, task.sim, task.num_envs
    names = gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
    hum_ids, tar_ids = task._humanoid_actor_ids, task._tar_actor_ids

    def set_h(t):
        rs = task._humanoid_root_states
        rs[:, 0:3] = md['root_pos'][t]; rs[:, 3:7] = md['root_rot'][t]
        rs[:, 7:10] = md['root_pos_vel'][t]; rs[:, 10:13] = md['root_rot_vel'][t]
        task._dof_pos[:] = md['dof_pos'][t]; task._dof_vel[:] = md['dof_pos_vel'][t]
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(hum_ids), n)
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._dof_state), gymtorch.unwrap_tensor(hum_ids), n)

    ts = task._target_states
    set_h(0); ts[:, 0:3] = md['obj_pos'][0]; ts[:, 3:7] = md['obj_rot'][0]; ts[:, 7:13] = 0
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(tar_ids), n)
    gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
    o0 = md['obj_pos'][0].cpu().numpy()
    first = [None] * n
    rows = []
    for t in range(1, frames):
        set_h(t); gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
        O = task._target_states[:, 0:3].cpu().numpy()
        V = task._target_states[:, 7:10].cpu().numpy()
        rows.append(np.linalg.norm(O[:, :2] - o0[:2], axis=-1))
        for e in range(n):
            if first[e] is None and rows[-1][e] > 0.03:
                bp = task._rigid_body_pos[e].cpu().numpy(); d = np.linalg.norm(bp - O[e], axis=-1)
                hcf = task._contact_forces[e].norm(dim=-1).cpu().numpy()
                first[e] = (t, O[e] - o0, np.linalg.norm(V[e]), [f"{names[i]} {d[i]*100:.0f}cm" for i in np.argsort(d)[:3]],
                            [names[i] for i in np.where(hcf > 1.0)[0]], task._tar_contact_forces[e].norm().item())
    rows = np.stack(rows)
    print("[hits] xy displacement (cm) per env at frames 1/5/10/20/39:")
    for e in range(n):
        print(f"[hits]  env {e}: " + " ".join(f"{rows[k, e]*100:5.1f}" for k in (0, 4, 9, 19, frames - 2)))
    for e in range(n):
        if first[e]:
            t, dd, v, near, touch, cf = first[e]
            print(f"[hits]  env {e}: moves at frame {t}: offset {np.round(dd*100,1)} cm, speed {v:.2f} m/s, obj force {cf:.0f} N, "
                  f"nearest {near}, humanoid bodies in contact {touch}")

def hand_gap(task, md, obj_name="cup", frames=(20, 30, 33, 36, 40, 50, 70, 90)):
    """Replay; distance from each hand's finger/palm bodies to the object MESH surface (reference object pose),
    i.e. does the reference hand pose actually touch the object with this hand model."""
    import numpy as np, torch
    from scipy.spatial.transform import Rotation as Rot
    V = np.array([[float(x) for x in l.split()[1:4]] for l in open(f"skillmimic/data/assets/obj/{obj_name}.obj") if l.startswith("v ")])
    V = V[:: max(1, len(V) // 8000)]
    names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
    hands = {"right": [i for i, nm in enumerate(names) if nm.startswith("jRight") and any(k in nm for k in ("Wrist", "CMC", "MCP", "PIP", "DIP", "IP"))],
             "left": [i for i, nm in enumerate(names) if nm.startswith("jLeft") and any(k in nm for k in ("Wrist", "CMC", "MCP", "PIP", "DIP", "IP"))]}
    # body origins of the 61 sim bodies == first 61 reference bodies (teleport check in replay)
    bp_all = md['body_pos'].cpu().numpy()
    op, orot = md['obj_pos'].cpu().numpy(), md['obj_rot'].cpu().numpy()
    print(f"[gap] min distance (cm) from hand body origins to the {obj_name} mesh at the reference pose (capsule radius 0.5 cm):")
    for t in frames:
        Rm = Rot.from_quat(orot[t]).as_matrix(); W = V @ Rm.T + op[t]
        out = []
        for h, ids in hands.items():
            d = np.linalg.norm(bp_all[t, ids][:, None] - W[None], axis=-1).min(-1)
            k = int(np.argmin(d)); out.append(f"{h}: {d.min()*100:4.1f} ({names[ids[k]]}), n<1.5cm {int((d < 0.015).sum())}")
        print(f"[gap]  frame {t:3d} obj z {op[t,2]:.3f} | " + " | ".join(out))


def he_phase(task, md):
    """How much clip-phase information the 3-d history embedding carries (the policy's only phase signal).
    Replay states; object dims from the reference so only the humanoid layout differs. Phase is decoded by
    nearest neighbour against the embeddings of pure reference windows; error in frames (30 fps)."""
    import numpy as np, torch
    from isaacgym import gymtorch
    from isaacgym.torch_utils import quat_rotate
    from utils import torch_utils
    from env.tasks.skillmimic_parahome_localhist import get_humanoid_obs, get_obj_obs
    gym, sim, n = task.gym, task.sim, task.num_envs
    hum_ids = task._humanoid_actor_ids
    T = min(md[k].shape[0] for k in ('root_pos', 'root_rot', 'root_pos_vel', 'root_rot_vel', 'dof_pos', 'dof_pos_vel'))
    bug, fix = [], []
    for t in range(T):
        rs = task._humanoid_root_states
        rs[:, 0:3] = md['root_pos'][t]; rs[:, 3:7] = md['root_rot'][t]
        rs[:, 7:10] = md['root_pos_vel'][t]; rs[:, 10:13] = md['root_rot_vel'][t]
        task._dof_pos[:] = md['dof_pos'][t]; task._dof_vel[:] = md['dof_pos_vel'][t]
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(hum_ids), n)
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._dof_state), gymtorch.unwrap_tensor(hum_ids), n)
        gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
        ho = task._compute_humanoid_obs(None)[0]
        rp, rr = task._rigid_body_pos[0:1, 0], task._rigid_body_rot[0:1, 0]
        pad = quat_rotate(torch_utils.calc_heading_quat_inv(rr), -rp).repeat(1, 10)[0]
        bug.append(torch.cat([ho[:211], task._dof_pos[0]]))
        fix.append(torch.cat([ho[:181], pad, task._dof_pos[0]]))
    ref_h = get_humanoid_obs(md['root_pos'], md['root_rot_3d'], md['body_pos'])[:T]
    ref_o = get_obj_obs(md['root_pos'], md['root_rot_3d'], md['obj_pos'])[:T]
    ref = torch.cat([ref_h, md['dof_pos'][:T].view(T, -1), ref_o], -1)
    bug = torch.cat([torch.stack(bug), ref_o], -1); fix = torch.cat([torch.stack(fix), ref_o], -1)
    # sim state lags the teleported reference by one physics step: compare sim row t with reference row t
    L = task.history_length
    print(f"[phase] |sim - ref| dims 181:211: buggy {(bug[:, 181:211] - ref[:, 181:211]).abs().mean():.4f}, "
          f"fixed {(fix[:, 181:211] - ref[:, 181:211]).abs().mean():.4f}; dims 0:181 {(fix[:, :181] - ref[:, :181]).abs().mean():.4f}")
    with torch.no_grad():
        E_ref = torch.stack([task.hist_encoder(ref[t - L:t][None])[0] for t in range(L, T)])
        def decode(src, k):
            errs = []
            for t in range(L, T):
                w = ref[t - L:t].clone()
                if k: w[-k:] = src[t - k:t]
                e = task.hist_encoder(w[None])[0]
                errs.append(abs(int(torch.argmin((E_ref - e).norm(dim=-1))) - (t - L)))
            return np.mean(errs), np.median(errs)
        for k in (5, 20, 60):
            mb, db = decode(bug, k); mf, df = decode(fix, k)
            print(f"[phase] last {k:2d} history frames from the simulator: phase error buggy layout mean {mb:5.1f} / median {db:4.1f} frames,"
                  f"  fixed layout mean {mf:5.1f} / median {df:4.1f}")
        print(f"[phase] (chance level for this clip ~{(T - L) / 3:.0f} frames)")


def grip(task, md, frame=60, steps=60, delta=0.6):
    """Can this hand hold the object at all? Start every env in the reference state of `frame` (object in hand),
    keep the root pinned at that pose, PD-drive all joints to the reference pose of that frame, and close the
    right/left finger joints by `delta` rad about one axis (x+, x-, y+, y-, z+, z-, none; one variant per env)."""
    import numpy as np, torch
    from isaacgym import gymtorch
    gym, sim, n = task.gym, task.sim, task.num_envs
    names = gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
    dof_names = gym.get_actor_dof_names(task.envs[0], task.humanoid_handles[0])
    fing = [i for i, d in enumerate(dof_names) if any(k in d for k in ("CMC", "MCP", "PIP", "DIP", "IP_"))]
    variants = [("none", None, 0)] + [(f"{ax}{'+' if sg > 0 else '-'}", ax, sg) for ax in "xyz" for sg in (1, -1)]
    ps = os.environ.get("PROBE_SHAPE")   # "<attr>:v1,v2,..." per-env object shape property instead of finger variants
    if ps:
        attr, vals = ps.split(":")
        vals = vals.split(",")
        variants = [(f"{attr[:4]}{v}", None, 0) for v in vals]
        for e, v in enumerate(vals):
            props = gym.get_actor_rigid_shape_properties(task.envs[e], task._target_handles[e])
            for pr in props:
                setattr(pr, attr, float(v))
            gym.set_actor_rigid_shape_properties(task.envs[e], task._target_handles[e], props)
    assert n >= len(variants), f"need {len(variants)} envs"
    hum_ids, tar_ids = task._humanoid_actor_ids, task._tar_actor_ids
    tgt = md['dof_pos'][frame].clone().repeat(n, 1)
    for e, (nm, ax, sg) in enumerate(variants):
        if ax:
            idx = [i for i in fing if dof_names[i].endswith("_" + ax)]
            tgt[e, idx] += sg * delta
    pd = (tgt - task._pd_action_offset) / task._pd_action_scale       # action that produces these targets
    def pin_root():
        rs = task._humanoid_root_states
        rs[:, 0:3] = md['root_pos'][frame]; rs[:, 3:7] = md['root_rot'][frame]; rs[:, 7:13] = 0
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(hum_ids), n)
    pin_root()
    task._dof_pos[:] = md['dof_pos'][frame]; task._dof_vel[:] = 0
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._dof_state), gymtorch.unwrap_tensor(hum_ids), n)
    ts = task._target_states
    ts[:, 0:3] = md['obj_pos'][frame]; ts[:, 3:7] = md['obj_rot'][frame]; ts[:, 7:13] = 0
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(tar_ids), n)
    z0 = md['obj_pos'][frame][2].item()
    Z = []
    gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
    from scipy.spatial.transform import Rotation as Rot
    up = np.array([0.71, -0.05, -0.71]); up /= np.linalg.norm(up)
    a_ = np.cross(up, [0, 1, 0]); a_ /= np.linalg.norm(a_); b_ = np.cross(up, a_)
    q = task._target_states[0, 3:7].cpu().numpy(); Rm = Rot.from_quat(q).as_matrix(); o = task._target_states[0, 0:3].cpu().numpy()
    qr = md['obj_rot'][frame].cpu().numpy(); Rr = Rot.from_quat(qr).as_matrix(); orf = md['obj_pos'][frame].cpu().numpy()
    hcf = task._contact_forces[0].norm(dim=-1).cpu().numpy()
    print("[grip] humanoid bodies with contact force > 1 N after 1 step: " + ", ".join(f"{names[i]} {hcf[i]:.0f}N" for i in np.argsort(-hcf) if hcf[i] > 1))
    print(f"[grip] after 1 step: object moved {np.linalg.norm(o - orf)*100:.1f} cm, rotated {np.degrees(np.arccos(np.clip((np.trace(Rr.T @ Rm) - 1) / 2, -1, 1))):.0f} deg; obj contact {task._tar_contact_forces[0].norm().item():.1f} N")
    for nm in ["jRightIP", "jRightSecondDIP", "jRightThirdDIP", "jRightFourthDIP", "jRightFifthDIP", "jRightWrist"]:
        i = names.index(nm)
        for tag, R_, o_, P in (("sim", Rm, o, task._rigid_body_pos[0, i].cpu().numpy()), ("ref", Rr, orf, md['body_pos'][frame, i].cpu().numpy())):
            l = R_.T @ (P - o_)
            print(f"[grip]   {tag} {nm:16s} h {l @ up:+.3f} r {np.hypot(l @ a_, l @ b_):.3f} az {np.degrees(np.arctan2(l @ b_, l @ a_)):+.0f}")
    for t in range(steps):
        task.pre_physics_step(pd.clamp(-1, 1))
        pin_root()
        gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
        Z.append(task._target_states[:, 2].cpu().numpy().copy())
    Z = np.stack(Z)
    print(f"[grip] frame {frame}: object start z {z0:.3f}; drop (cm) after 5/15/30/{steps} steps, root pinned, PD to reference pose")
    for e, (nm, ax, sg) in enumerate(variants):
        print(f"[grip]   fingers {nm:5s}: " + " / ".join(f"{(z0 - Z[k, e]) * 100:5.1f}" for k in (4, 14, 29, steps - 1)))


if __name__ == "__main__":
    main()
