"""Can this environment physically hold the object the way the demonstration does? (GPU, ~1 min per frame)

    python environment_iteration/src/probes/grip.py <iteration> --frames 90,120 [--variants "base;obj_friction=2;..."]
    python environment_iteration/src/probes/grip.py <iteration> --replay 40:147     # perfect imitator, whole clip

For every env: start in the reference state of a clip frame (humanoid joints + root + object exactly as in the
demonstration), pin the root there, and PD-drive every joint to that frame's reference pose (= a perfect
imitator that does not move). Then simulate `--steps` steps and measure where the object goes. If the object
falls out even here, no policy that imitates the reference hand pose can hold it: the environment (contact model,
geometry, mass, ...) is the problem, not the learning.

Each env runs one variant, applied AT ACTOR CREATION (shape / body / dof properties set after prepare_sim are not
reliably applied on the GPU pipeline; the experiments/probe/env_probe.py PROBE_SHAPE results were identical for all
values for that reason). Variant keys (combine with ',' inside one variant, separate variants with ';'):
  obj_friction=F  obj_torsion=F  obj_rolling=F  obj_restitution=F    object shape properties
  hand_friction=F hand_torsion=F                                     shapes of all hand bodies (wrists and fingers)
  obj_mass=KG                                                        object mass (inertia recomputed)
  finger_kp=X finger_kd=X                                            PD gains of finger joints (asset: 500 / 500)
  close=RAD                                                          close all finger joints by RAD about x (+/-)
Printed per variant: object drop (cm) and object displacement relative to the holding wrist after 5/15/30/steps,
HELD = after settling (first 5 steps) the object stays within 2 cm of the wrist and drops < 15 cm in total;
before that, the gap between the simulated fingertips (reference joint angles through this model's kinematics)
and the reference fingertips, which tells whether the model can even place its fingers where the human's were.
"""
import argparse
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)
import iterlib as L  # noqa: E402

FINGER_KEYS = ("CMC", "MCP", "PIP", "DIP", "IP_")
HAND_BODY_KEYS = ("Wrist", "CMC", "MCP", "PIP", "DIP", "IP")


def parse_variants(spec):
    out = []
    for v in spec.split(";"):
        v = v.strip()
        d = {}
        if v and v != "base":
            for kv in v.split(","):
                k, x = kv.split("=")
                d[k.strip()] = float(x)
        out.append((v or "base", d))
    return out


class VariantHook:
    """Duck-typed 'cameras' object for rollout.Env: install() runs before the task is built."""

    def __init__(self, variants):
        self.variants = variants

    def install(self, task_cls):
        hook = self
        orig_env, orig_tar = task_cls._build_env, task_cls._build_target

        def _build_target(self_, env_id, env_ptr, _o=orig_tar):
            _o(self_, env_id, env_ptr)
            d = hook.variants[env_id % len(hook.variants)][1]
            h = self_.target_handle
            props = self_.gym.get_actor_rigid_shape_properties(env_ptr, h)
            for p in props:
                if "obj_friction" in d: p.friction = d["obj_friction"]
                if "obj_torsion" in d: p.torsion_friction = d["obj_torsion"]
                if "obj_rolling" in d: p.rolling_friction = d["obj_rolling"]
                if "obj_restitution" in d: p.restitution = d["obj_restitution"]
            self_.gym.set_actor_rigid_shape_properties(env_ptr, h, props)
            if "obj_mass" in d:
                bp = self_.gym.get_actor_rigid_body_properties(env_ptr, h)
                bp[0].mass = d["obj_mass"]
                self_.gym.set_actor_rigid_body_properties(env_ptr, h, bp, True)

        def _build_env(self_, env_id, env_ptr, humanoid_asset, _o=orig_env):
            _o(self_, env_id, env_ptr, humanoid_asset)
            d = hook.variants[env_id % len(hook.variants)][1]
            h = self_.humanoid_handles[-1]
            g = self_.gym
            if any(k in d for k in ("hand_friction", "hand_torsion")):
                names = g.get_actor_rigid_body_names(env_ptr, h)
                ranges = g.get_actor_rigid_body_shape_indices(env_ptr, h)
                props = g.get_actor_rigid_shape_properties(env_ptr, h)
                for bi, nm in enumerate(names):
                    if any(k in nm for k in HAND_BODY_KEYS):
                        for si in range(ranges[bi].start, ranges[bi].start + ranges[bi].count):
                            if "hand_friction" in d: props[si].friction = d["hand_friction"]
                            if "hand_torsion" in d: props[si].torsion_friction = d["hand_torsion"]
                g.set_actor_rigid_shape_properties(env_ptr, h, props)
            if "finger_kp" in d or "finger_kd" in d:
                dp = g.get_actor_dof_properties(env_ptr, h)
                for i, nm in enumerate(g.get_actor_dof_names(env_ptr, h)):
                    if any(k in nm for k in FINGER_KEYS):
                        if "finger_kp" in d: dp["stiffness"][i] = d["finger_kp"]
                        if "finger_kd" in d: dp["damping"][i] = d["finger_kd"]
                g.set_actor_dof_properties(env_ptr, h, dp)
        task_cls._build_target = _build_target
        task_cls._build_env = _build_env

    def attach(self, env):
        pass


SUMMARY = {}   # variant -> [envs held, envs tried] over all frames


def hold(env, variants, frame, steps, hand):
    import numpy as np
    import torch
    from isaacgym import gymtorch
    task, md = env.task, env.md.hoi_data_dict[0]
    gym, sim, n = task.gym, task.sim, task.num_envs
    names = gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
    dof_names = gym.get_actor_dof_names(task.envs[0], task.humanoid_handles[0])
    hum_ids, tar_ids = task._humanoid_actor_ids, task._tar_actor_ids
    tgt = md["dof_pos"][frame].clone().repeat(n, 1)
    for e in range(n):
        d = variants[e % len(variants)][1]
        if "close" in d:
            idx = [i for i, nm in enumerate(dof_names) if nm.startswith(f"j{hand}") and
                   any(k in nm for k in FINGER_KEYS) and nm.endswith("_x")]
            tgt[e, idx] += d["close"]
    act = ((tgt - task._pd_action_offset) / task._pd_action_scale).clamp(-1, 1)

    def pin_root():
        rs = task._humanoid_root_states
        rs[:, 0:3] = md["root_pos"][frame]; rs[:, 3:7] = md["root_rot"][frame]; rs[:, 7:13] = 0
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states),
                                                gymtorch.unwrap_tensor(hum_ids), n)
    pin_root()
    task._dof_pos[:] = md["dof_pos"][frame]; task._dof_vel[:] = 0
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._dof_state), gymtorch.unwrap_tensor(hum_ids), n)
    ts = task._target_states
    ts[:, 0:3] = md["obj_pos"][frame]; ts[:, 3:7] = md["obj_rot"][frame]; ts[:, 7:13] = 0
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states),
                                            gymtorch.unwrap_tensor(tar_ids), n)
    gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
    tips = [names.index(f"j{hand}{t}") for t in ("IP", "SecondDIP", "ThirdDIP", "FourthDIP", "FifthDIP")]
    wrist = names.index(f"j{hand}Wrist")
    sim_tips = task._rigid_body_pos[0, tips].cpu().numpy()
    ref_tips = md["body_pos"][frame, tips].cpu().numpy()
    gap = np.linalg.norm(sim_tips - ref_tips, axis=-1) * 100
    print(f"[grip] frame {frame}: {hand.lower()} fingertip position, model (reference joint angles) vs reference data "
          f"(cm): " + " ".join(f"{g:.1f}" for g in gap) + "  (thumb..little)")
    o0 = task._target_states[:, 0:3].clone()
    rel0 = (o0 - task._rigid_body_pos[:, wrist]).cpu().numpy()
    z0 = md["obj_pos"][frame][2].item()
    Z, REL = [], []
    for t in range(steps):
        task.pre_physics_step(act)
        pin_root()
        gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
        Z.append(task._target_states[:, 2].cpu().numpy().copy())
        REL.append((task._target_states[:, 0:3] - task._rigid_body_pos[:, wrist]).cpu().numpy() - rel0)
    Z, REL = np.stack(Z), np.stack(REL)
    ks = [k for k in (4, 14, 29, steps - 1) if k < steps]
    print(f"[grip] frame {frame}: object start z {z0:.3f}. per variant: drop (cm) | slip vs {hand.lower()} wrist (cm) "
          f"after {'/'.join(str(k + 1) for k in ks)} steps, root pinned, PD to the reference pose")
    rows = {}
    for e in range(n):
        nm = variants[e % len(variants)][0]
        rows.setdefault(nm, []).append(e)
    for nm, es in rows.items():
        dr = [np.mean([(z0 - Z[k, e]) * 100 for e in es]) for k in ks]
        sl = [np.mean([np.linalg.norm(REL[k, e]) * 100 for e in es]) for k in ks]
        # held = after the first 5 steps (settling into the grip) the object stays within 2 cm of where it settled
        held = "HELD" if abs(sl[-1] - sl[0]) < 2.0 and dr[-1] < 15 else "lost"
        n_held = sum(1 for e in es if abs(np.linalg.norm(REL[-1, e]) - np.linalg.norm(REL[min(4, steps - 1), e])) * 100
                     < 2.0 and (z0 - Z[-1, e]) * 100 < 15)
        print(f"[grip]   {nm:40s} drop " + " / ".join(f"{x:5.1f}" for x in dr) + "  | slip "
              + " / ".join(f"{x:5.1f}" for x in sl) + f"  {held}  envs held {n_held}/{len(es)}")
        SUMMARY.setdefault(nm, [0, 0])
        SUMMARY[nm][0] += n_held
        SUMMARY[nm][1] += len(es)
    return rows


def replay(env, variants, f0, f1, detail=None):
    """Perfect imitator: from reference frame f0 to f1, every step pin the root to the reference root of that frame
    and PD-drive all joints to that frame's reference pose (velocities set to the reference); the object simulates
    freely from its frame-f0 reference pose. Prints, per variant, how long the object stays within 10 cm of its
    reference (the task-success rule) and its error at selected frames."""
    import numpy as np
    from isaacgym import gymtorch
    task, md = env.task, env.md.hoi_data_dict[0]
    gym, sim, n = task.gym, task.sim, task.num_envs
    hum_ids, tar_ids = task._humanoid_actor_ids, task._tar_actor_ids

    def set_root(fr):
        rs = task._humanoid_root_states
        rs[:, 0:3] = md["root_pos"][fr]; rs[:, 3:7] = md["root_rot"][fr]
        rs[:, 7:10] = md["root_pos_vel"][fr]; rs[:, 10:13] = md["root_rot_vel"][fr]
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states),
                                                gymtorch.unwrap_tensor(hum_ids), n)
    set_root(f0)
    task._dof_pos[:] = md["dof_pos"][f0]; task._dof_vel[:] = md["dof_pos_vel"][f0]
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._dof_state), gymtorch.unwrap_tensor(hum_ids), n)
    ts = task._target_states
    ts[:, 0:3] = md["obj_pos"][f0]; ts[:, 3:7] = md["obj_rot"][f0]
    ts[:, 7:10] = md["obj_pos_vel"][f0]; ts[:, 10:13] = md["obj_rot_vel"][f0]   # object moves with the hand at f0
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(task._root_states),
                                            gymtorch.unwrap_tensor(tar_ids), n)
    gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
    names = gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
    lh = [i for i, nm in enumerate(names) if nm.startswith("jLeft") and any(k in nm for k in HAND_BODY_KEYS)]
    rh = [i for i, nm in enumerate(names) if nm.startswith("jRight") and any(k in nm for k in HAND_BODY_KEYS)]
    q0 = md["obj_rot"][f0].cpu().numpy()

    def ang(a, b):
        return float(np.degrees(2 * np.arccos(min(1.0, abs(float((a * b).sum()))))))
    err = []
    if detail:
        print("[replay-detail] env 0: ref frame | book z sim/ref | xy moved sim/ref (cm) | tilt from start sim/ref | "
              "rot err vs ref | contact force: book total, left hand, right hand (N)")
    for fr in range(f0 + 1, f1 + 1):
        act = ((md["dof_pos"][fr] - task._pd_action_offset) / task._pd_action_scale).clamp(-1, 1).repeat(n, 1)
        task.pre_physics_step(act)
        set_root(fr)
        gym.simulate(sim); gym.fetch_results(sim, True); task._refresh_sim_tensors()
        err.append((task._target_states[:, 0:3] - md["obj_pos"][fr]).norm(dim=-1).cpu().numpy())
        if detail and fr % 2 == 0 and detail[0] <= fr <= detail[1]:
            o = task._target_states[0].cpu().numpy(); ro = md["obj_pos"][fr].cpu().numpy()
            rq = md["obj_rot"][fr].cpu().numpy()
            cf = task._contact_forces[0].norm(dim=-1).cpu().numpy()
            print(f"[replay-detail]   {fr:3d} | {o[2]:.3f}/{ro[2]:.3f} | "
                  f"{np.linalg.norm(o[:2] - md['obj_pos'][f0][:2].cpu().numpy()) * 100:4.1f}/"
                  f"{np.linalg.norm(ro[:2] - md['obj_pos'][f0][:2].cpu().numpy()) * 100:4.1f} | "
                  f"{ang(o[3:7], q0):4.0f}/{ang(rq, q0):4.0f} | {ang(o[3:7], rq):4.0f} | "
                  f"{task._tar_contact_forces[0].norm().item():6.1f} {cf[lh].sum():6.1f} {cf[rh].sum():6.1f}")
    err = np.stack(err)                                        # (frames, envs); row k = reference frame f0 + 1 + k
    frames = np.arange(f0 + 1, f1 + 1)
    show = [f for f in (60, 66, 70, 80, 90, 100, 120, 140) if f0 < f <= f1]
    rows = {}
    for e in range(n):
        rows.setdefault(variants[e % len(variants)][0], []).append(e)
    print(f"[replay] reference frames {f0}->{f1}, perfect imitator (root pinned to the reference, joints PD to it)")
    print(f"[replay]   {'variant':32s} lost-track frame (median, 10 cm) | object error (cm) at frames "
          + " ".join(map(str, show)))
    for nm, es in rows.items():
        lost = []
        for e in es:
            bad = np.nonzero(err[:, e] > 0.10)[0]
            lost.append(int(frames[bad[0]]) if bad.size else f1 + 1)
        med = [np.median(err[frames == f, es]) * 100 for f in show]
        print(f"[replay]   {nm:32s} {np.median(lost):5.0f} (range {min(lost)}-{max(lost)}) | "
              + " ".join(f"{x:5.1f}" for x in med))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration")
    p.add_argument("--frames", default="90", help="reference (motion) frames, comma-separated")
    p.add_argument("--variants", default="base")
    p.add_argument("--repeat", type=int, default=2, help="envs per variant")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--hand", default="Left", choices=["Left", "Right"])
    p.add_argument("--asset", default="", help="humanoid MJCF under skillmimic/data/assets instead of the "
                   "iteration's (same bodies / dofs), e.g. one written by src/tools/hand_asset.py")
    p.add_argument("--motion", default="", help="motion folder under skillmimic/data/motions/ParaHome instead of "
                   "the iteration's (e.g. one written by src/tools/contact_reference.py)")
    p.add_argument("--set", default="", help="env.yaml edits for this probe, e.g. "
                   "'sim.substeps=4;sim.physx.contact_offset=0.005' (a scratch copy of env.yaml is used)")
    p.add_argument("--book_urdf", default="", help="object URDF under skillmimic/data/assets/obj instead of the "
                   "iteration's dynamicUrdf (a scratch copy of env.yaml is used)")
    p.add_argument("--replay", default="", help="F0:F1 run the perfect-imitator replay over these reference frames "
                   "instead of the static holds")
    p.add_argument("--detail", default="", help="with --replay: F0:F1 print a per-frame trace of env 0")
    p.add_argument("--checkpoint", default="", help="any checkpoint with this iteration's obs size (default: the "
                   "iteration's newest, else fix-1 epoch 3000)")
    a = p.parse_args()
    variants = parse_variants(a.variants)
    ck = a.checkpoint
    if not ck:
        c = L.checkpoints(L.paths(a.iteration)["run_dir"]) or L.checkpoints(L.paths("fix-1")["run_dir"])
        ck = [x for e, x in c if e == 3000][0] if any(e == 3000 for e, _ in c) else c[-1][1]
    import rollout as RO
    ov = {"asset": a.asset} if a.asset else {}
    if a.motion:
        ov["motion_file"] = f"skillmimic/data/motions/ParaHome/{a.motion}"
        print(f"[grip] motion: {a.motion}")
    if a.set:
        import tempfile
        import yaml
        cfg = yaml.safe_load(open(os.path.join(L.REPO, ov.get("cfg_env") or L.load(a.iteration)["cfg_env"])))
        for kv in a.set.split(";"):
            k, v = kv.split("=")
            node = cfg
            ks = k.strip().split(".")
            for kk in ks[:-1]:
                node = node[kk]
            node[ks[-1]] = yaml.safe_load(v)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.environ.get("TMPDIR", "/tmp"))
        yaml.safe_dump(cfg, tmp); tmp.close()
        ov["cfg_env"] = tmp.name
        print(f"[grip] env edits: {a.set}")
    if a.book_urdf:
        import re
        import tempfile
        y = open(os.path.join(L.REPO, ov.get("cfg_env") or L.load(a.iteration)["cfg_env"])).read()
        y, k = re.subn(r"(?m)^(\s*dynamicUrdf:\s*)\S+", lambda m: m.group(1) + a.book_urdf, y)
        assert k == 1, "env.yaml has no dynamicUrdf line"
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.environ.get("TMPDIR", "/tmp"))
        tmp.write(y); tmp.close()
        ov["cfg_env"] = tmp.name
        print(f"[grip] object: {a.book_urdf}")
    env = RO.Env(a.iteration, len(variants) * a.repeat, "det", 300, cameras=VariantHook(variants),
                 first_checkpoint=ck, overrides=ov or None)
    if a.asset:
        print(f"[grip] humanoid asset: {a.asset}")
    if a.replay:
        f0, f1 = (int(x) for x in a.replay.split(":"))
        return replay(env, variants, f0, f1, tuple(int(x) for x in a.detail.split(":")) if a.detail else None)
    for fr in [int(x) for x in a.frames.split(",")]:
        hold(env, variants, fr, a.steps, a.hand)
    print(f"[grip] SUMMARY over frames {a.frames}" + (f", asset {a.asset}" if a.asset else "") + ": "
          + "; ".join(f"{nm} held {h}/{n}" for nm, (h, n) in SUMMARY.items()))


if __name__ == "__main__":
    main()
