"""Feet, body, hands and book on one time axis (ENV_FIX_PLAN.md E1 / E2). GPU, ~1-2 min per configuration.

    python environment_iteration/src/probes/gait_grip.py fix-4 --epoch 3000 --start 2,85 --seeds 101,102,103
    python environment_iteration/src/probes/gait_grip.py fix-4 --start 2 --assist 0,0.5 --assist_from 75

Rolls out an iteration's policy (stoch, clean start at reference frame `start`, like policy_starts.py: body AND book
in the reference state) and records, per control step, what the standard trace does not have:

  feet   net contact force of jLeftAnkle / jLeftBallFoot / jRightAnkle / jRightBallFoot (body indices looked up by
         name at runtime), their heights and horizontal velocities
  body   root (pelvis) position / height, leg DOF PD target vs actual, travel along the reference walk direction
  hands  both wrists, fingertip contact force per hand, book pose in the left-wrist frame
  book   pose / velocity, net contact force, the mass / COM the simulator uses

and prints, per configuration, the median reference frame of each event (in the order of the plan's timeline):

  R release -> L-book slip -> pelvis drop -> first foot lift-off -> first step (>= 10 cm) -> body progress
  (>= 0.3 m along the reference walk) -> book drop

plus window statistics over reference frames `--window` (default 85-145, the reference carry walk).

E2 (`--assist a`): an upward force a * m_book * |g| on the book's centre of mass (world frame, applied before every
simulate), switched on at reference frame `--assist_from` with a `--assist_ramp`-frame linear ramp. a = 0 runs the
same code path with a zero force. Global gravity, the book's mass and inertia, and the hand are unchanged.
Diagnostic only: an assisted rollout is not a task result.

Net contact forces are the sum over all contacts of a body (no contact pairs). For the feet that is the floor,
except when a foot touches the other leg or the desk, which this probe does not separate.
"""
import argparse
import json
import os
import subprocess
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

FOOT_NAMES = ["jLeftAnkle", "jLeftBallFoot", "jRightAnkle", "jRightBallFoot"]
LEG_NAMES = ["jRightHip", "jRightKnee", "jRightAnkle", "jRightBallFoot",
             "jLeftHip", "jLeftKnee", "jLeftAnkle", "jLeftBallFoot"]
CONTACT_N = 20.0        # foot body in contact above this net force (body weight ~ 700 N; noise in free air < 1 N)
LIFT_FRAMES = 3         # a foot is lifted when both its bodies are below CONTACT_N for this many frames
STEP_M = 0.10           # a lift-off..touchdown with this much horizontal ankle displacement is a step
SLIP_MS = 0.10          # supporting ball-foot body moving faster than this horizontally = slip


def body_index(task, name):
    from isaacgym import gymapi
    i = task.gym.find_actor_rigid_body_index(task.envs[0], task.humanoid_handles[0], name, gymapi.DOMAIN_ACTOR)
    if i < 0:
        raise SystemExit(f"body {name} not found in the humanoid asset")
    return int(i)


def run_one(a, start, assist):
    import iterlib as L
    import rollout as RO

    run_dir = L.paths(a.iteration)["run_dir"]
    ck = L.checkpoint_at(run_dir, a.epoch)
    if not ck:
        raise SystemExit(f"no checkpoint e{a.epoch} in {run_dir}")
    horizon = a.horizon if a.horizon > 0 else None
    env = RO.Env(a.iteration, a.num_envs, "stoch", horizon or 150, start_frame=start, seed=a.seeds[0],
                 first_checkpoint=ck)
    from isaacgym import gymapi, gymtorch   # after RO.Env: isaacgym must be imported before torch
    import numpy as np
    import torch
    from utils import torch_utils
    task, player = env.task, env.player
    clip = env.clip_frames
    if horizon is None:
        horizon = clip - start - 1
    E = a.num_envs
    feet = [body_index(task, n) for n in FOOT_NAMES]
    legs = [body_index(task, n) for n in LEG_NAMES]
    leg_dofs = torch.tensor(sorted(d for b in legs for d in range(3 * (b - 1), 3 * (b - 1) + 3)), device=task.device)
    nb = task.num_bodies
    bodies_per_env = task._rigid_body_state.shape[0] // E
    props = task.gym.get_actor_rigid_body_properties(task.envs[0], task._target_handles[0])
    book_mass = float(props[0].mass)
    book_com = [props[0].com.x, props[0].com.y, props[0].com.z]
    g = task.sim_params.gravity
    g_mag = float(np.sqrt(g.x ** 2 + g.y ** 2 + g.z ** 2))
    print(f"[gait] bodies {dict(zip(FOOT_NAMES, feet))} num_bodies {nb} per env {bodies_per_env} | book mass "
          f"{book_mass:.3f} kg com {np.round(book_com, 4).tolist()} | g {g_mag:.2f} | assist {assist} from ref "
          f"{a.assist_from} ramp {a.assist_ramp}", flush=True)

    force = torch.zeros(E * bodies_per_env, 3, device=task.device, dtype=torch.float)
    book_rows = torch.arange(E, device=task.device) * bodies_per_env + nb   # book = first body after the humanoid
    rec = {"pd_tar": None, "assist_n": torch.zeros(E, device=task.device)}
    orig_pre = task.pre_physics_step

    def pre_physics_step(actions):
        orig_pre(actions)
        rec["pd_tar"] = task._action_to_pd_targets(task.actions)[:, leg_dofs].clone()
        ref_f = (env.md.envid2sframe.to(task.device) + task.progress_buf).float()
        ramp = torch.clamp((ref_f - a.assist_from) / max(a.assist_ramp, 1), 0.0, 1.0)
        fz = assist * book_mass * g_mag * ramp
        force.zero_()
        force[book_rows, 2] = fz
        rec["assist_n"] = fz
        task.gym.apply_rigid_body_force_tensors(task.sim, gymtorch.unwrap_tensor(force), None, gymapi.ENV_SPACE)
    task.pre_physics_step = pre_physics_step

    ref = env.md.hoi_data_dict[0]
    ref_root = ref["root_pos"].detach().cpu().numpy()
    walk0, walk1 = 85, clip - 1
    wdir = ref_root[walk1, :2] - ref_root[walk0, :2]
    wdir = wdir / (np.linalg.norm(wdir) + 1e-9)

    out = {}
    for sd in a.seeds:
        player.restore(ck)
        torch.manual_seed(sd); np.random.seed(sd % (2 ** 32))
        import random as _r; _r.seed(sd)
        obs = player.env_reset()
        player.get_batch_size(obs["obs"], 1)
        alive = torch.ones(E, dtype=torch.bool, device=task.device)
        keys = ["ref_frame", "alive", "root", "root_vel", "ref_root", "foot_f", "foot_z", "foot_vxy", "foot_pos",
                "leg_tar", "leg_dof", "wrist_l", "wrist_r", "tip_f_l", "tip_f_r", "obj", "obj_rot", "obj_vel",
                "ref_obj", "obj_f", "book_in_l", "rot_in_l", "assist_n", "reward"]
        T = {k: [] for k in keys}
        for t in range(horizon):
            action = player.get_action(obs, False)
            obs_raw, _, done, inf = player.env_step(player.env, action)
            obs = player.obs_to_torch(obs_raw)
            fell_now = inf["terminate"].bool() & alive
            rb, rr, rv = task._rigid_body_pos, task._rigid_body_rot, task._rigid_body_vel
            cf = task._contact_forces
            obj = task._target_states[:, 0:3]
            orot = task._target_states[:, 3:7]
            wl, wlr = rb[:, RO.WRIST_L], rr[:, RO.WRIST_L]
            inv = torch_utils.quat_conjugate(wlr)
            vals = dict(
                ref_frame=env.md.envid2sframe.to(task.device) + task.progress_buf, alive=alive & ~fell_now,
                root=rb[:, 0], root_vel=rv[:, 0], ref_root=task._curr_ref_obs[:, 0:3],
                foot_f=cf[:, feet].norm(dim=-1), foot_z=rb[:, feet, 2], foot_vxy=rv[:, feet, :2].norm(dim=-1),
                foot_pos=rb[:, feet], leg_tar=rec["pd_tar"], leg_dof=task._dof_pos[:, leg_dofs],
                wrist_l=wl, wrist_r=rb[:, RO.WRIST_R],
                tip_f_l=cf[:, env.hand_l].norm(dim=-1).sum(-1), tip_f_r=cf[:, env.hand_r].norm(dim=-1).sum(-1),
                obj=obj, obj_rot=orot, obj_vel=task._target_states[:, 7:10],
                ref_obj=task._curr_ref_obs[:, RO.REF_OBJ_POS], obj_f=task._tar_contact_forces.norm(dim=-1),
                book_in_l=torch_utils.quat_rotate(inv, obj - wl), rot_in_l=torch_utils.quat_mul(inv, orot),
                assist_n=rec["assist_n"].expand(E) if rec["assist_n"].dim() == 0 else rec["assist_n"],
                reward=task.rew_buf)
            for k in keys:
                T[k].append(vals[k].detach().clone().float() if vals[k].dtype != torch.bool else vals[k].clone())
            alive = alive & ~fell_now & ~(done.bool() & (t < horizon - 1))
        out[sd] = {k: torch.stack(v).cpu().numpy() for k, v in T.items()}
    kb = list(task.cfg["env"]["keyBodies"])
    rk = ref["key_body_pos"].detach().cpu().numpy().reshape(clip, len(kb), 3)
    ref_ankles = rk[:, [kb.index("jLeftAnkle"), kb.index("jRightAnkle")]]
    return out, dict(book_mass=book_mass, g=g_mag, wdir=wdir.tolist(), ref_root=ref_root, clip=clip,
                     ref_ankles=ref_ankles, obj_z0=float(ref["obj_pos"][2, 2]))


def first(mask, frames, need=1):
    """First reference frame from which mask holds for `need` consecutive steps (nan if never)."""
    import numpy as np
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= need:
            return float(frames[i - need + 1])
    return np.nan


def episode_events(D, e, meta, a):
    import numpy as np
    fr = D["ref_frame"][:, e].astype(float)
    al = D["alive"][:, e]
    n = int(al.sum()) if not al.all() else len(fr)
    sl = slice(0, max(n, 1))
    fr = fr[sl]
    ff = D["foot_f"][sl, e]                     # (T, 4): L ankle, L ball, R ankle, R ball
    contact_l = (ff[:, 0] > CONTACT_N) | (ff[:, 1] > CONTACT_N)
    contact_r = (ff[:, 2] > CONTACT_N) | (ff[:, 3] > CONTACT_N)
    obj, wl, wr = D["obj"][sl, e], D["wrist_l"][sl, e], D["wrist_r"][sl, e]
    dl, dr = np.linalg.norm(obj - wl, axis=-1), np.linalg.norm(obj - wr, axis=-1)
    ev = {}
    # right release (estimated): right wrist > 0.25 m from the book centre for 5 frames after the book is up. Fingertip
    # net force cannot be used: the clenched fingers press on each other (0-200 N with the wrist 1 m from the book)
    up = obj[:, 2] > meta["obj_z0"] + 0.02
    ev["r_release"] = first(up & (dr > 0.25), fr, 5)
    # left-hand-book slip: book position in the left-wrist frame moves > 5 cm from its value 3 frames after release
    bl = D["book_in_l"][sl, e]
    rel = ev["r_release"]
    if np.isfinite(rel):
        i0 = min(int(np.argmax(fr >= rel)) + 3, len(fr) - 1)
        dev = np.linalg.norm(bl - bl[i0], axis=-1)
        m = (np.arange(len(fr)) >= i0) & (dev > 0.05) & (dl < 0.3)
        ev["l_slip"] = first(m, fr, 3)
    else:
        ev["l_slip"] = np.nan
    root = D["root"][sl, e]
    ev["pelvis_drop"] = first(root[:, 2] < 0.78, fr, 3)
    # foot lift-off after the walk window start (reference frame 80: earlier shuffles are the pickup)
    after = fr >= 80
    lift_l, lift_r = ~contact_l, ~contact_r
    ev["lift_l"] = first(after & lift_l, fr, LIFT_FRAMES)
    ev["lift_r"] = first(after & lift_r, fr, LIFT_FRAMES)
    ev["first_lift"] = np.nanmin([ev["lift_l"], ev["lift_r"]]) if np.isfinite([ev["lift_l"], ev["lift_r"]]).any() else np.nan
    # steps: lift-off .. touchdown with >= STEP_M horizontal ankle displacement
    fp = D["foot_pos"][sl, e]
    steps = []
    for side, c, idx in (("L", contact_l, 0), ("R", contact_r, 2)):
        i = 0
        while i < len(fr):
            if after[i] and not c[i]:
                j = i
                while j < len(fr) and not c[j]:
                    j += 1
                if j - i >= LIFT_FRAMES and j < len(fr):
                    d = np.linalg.norm(fp[j, idx, :2] - fp[max(i - 1, 0), idx, :2])
                    along = float(np.dot(fp[j, idx, :2] - fp[max(i - 1, 0), idx, :2], meta["wdir"]))
                    steps.append((fr[i], fr[j], side, d, along))
                i = j + 1
            else:
                i += 1
    steps.sort()
    big = [s for s in steps if s[3] >= STEP_M]
    ev["first_step"] = big[0][1] if big else np.nan
    ev["n_steps"] = len(big)
    ev["n_lifts"] = len(steps)
    ev["step_len_mean"] = float(np.mean([s[3] for s in big])) if big else 0.0
    i85 = np.argmax(fr >= 85) if (fr >= 85).any() else 0
    prog = (root[:, :2] - root[i85, :2]) @ np.asarray(meta["wdir"])
    ev["progress"] = first((fr >= 85) & (prog >= 0.3), fr, 1)
    ev["travel_along"] = float(prog[-1]) if len(prog) else 0.0
    drop = up & (np.minimum(dl, dr) > 0.3)
    ev["drop"] = first(drop | (obj[:, 2] < 0.5), fr, 3)
    # window statistics
    w = (fr >= a.window[0]) & (fr <= a.window[1])
    ev["win_n"] = int(w.sum())
    if w.any():
        ball_v = D["foot_vxy"][sl, e][:, [1, 3]]
        ball_c = ff[:, [1, 3]] > CONTACT_N
        sv = ball_v[w][ball_c[w]]
        ev["slip_frac"] = float((sv > SLIP_MS).mean()) if sv.size else np.nan
        ev["slip_v_med"] = float(np.median(sv)) if sv.size else np.nan
        fl = ff[:, 0] + ff[:, 1]
        frt = ff[:, 2] + ff[:, 3]
        share = fl / np.maximum(fl + frt, 1e-6)
        ev["load_l_std"] = float(np.std(share[w]))           # weight shifting between the feet
        ev["both_contact"] = float((contact_l & contact_r)[w].mean())
        ev["no_contact"] = float((~contact_l & ~contact_r)[w].mean())
        ev["leg_err"] = float(np.abs(D["leg_tar"][sl, e] - D["leg_dof"][sl, e])[w].mean())
        ev["leg_tar_vel"] = float(np.abs(np.diff(D["leg_tar"][sl, e], axis=0))[w[1:]].mean()) if w[1:].any() else np.nan
        ev["pelvis_z"] = float(np.median(root[w, 2]))
        ev["stance_w"] = float(np.median(np.linalg.norm(fp[w, 0, :2] - fp[w, 2, :2], axis=-1)))
        ev["root_v"] = float(np.median(np.linalg.norm(D["root_vel"][sl, e][w, :2], axis=-1)))
        ev["book_err"] = float(np.median(np.linalg.norm(obj[w] - D["ref_obj"][sl, e][w], axis=-1)))
        ev["held_l"] = float((dl[w] < 0.2).mean())
        ev["reward"] = float(D["reward"][sl, e][w].mean())
    return ev, steps


def ref_gait(meta, a):
    """Reference ankles in the window: swing = ankle > 3 cm above its window minimum; step = swing with >= STEP_M."""
    import numpy as np
    A = meta["ref_ankles"]
    lo, hi = a.window[0], min(a.window[1], len(A) - 1)
    txt = []
    for j, side in enumerate("LR"):
        z = A[lo:hi + 1, j, 2]
        sw = z > z.min() + 0.03
        steps, i = [], 0
        while i < len(z):
            if sw[i]:
                k = i
                while k < len(z) and sw[k]:
                    k += 1
                d = np.linalg.norm(A[lo + min(k, len(z) - 1), j, :2] - A[lo + max(i - 1, 0), j, :2])
                steps.append((lo + i, lo + k, round(float(d), 2)))
                i = k
            else:
                i += 1
        txt.append(f"{side}: " + ", ".join(f"{s}-{e} {d}m" for s, e, d in steps if d >= STEP_M))
    root = meta["ref_root"]
    along = float((root[hi, :2] - root[lo, :2]) @ np.asarray(meta["wdir"]))
    return f"reference ref {lo}-{hi}: root travel along walk {along:.2f} m; steps " + " | ".join(txt)


def summarize(tag, out, meta, a):
    import numpy as np
    rows = []
    for sd, D in out.items():
        for e in range(D["alive"].shape[1]):
            ev, _ = episode_events(D, e, meta, a)
            ev["seed"] = sd
            rows.append(ev)
    keys_ev = ["r_release", "l_slip", "pelvis_drop", "first_lift", "first_step", "progress", "drop"]
    keys_w = ["n_lifts", "n_steps", "step_len_mean", "travel_along", "slip_frac", "slip_v_med", "load_l_std",
              "both_contact", "no_contact", "leg_err", "leg_tar_vel", "pelvis_z", "stance_w", "root_v", "book_err",
              "held_l", "reward"]
    n = len(rows)
    print(f"[gait] {ref_gait(meta, a)}", flush=True)
    parts = []
    for k in keys_ev:
        v = np.array([r[k] for r in rows], float)
        ok = np.isfinite(v)
        parts.append(f"{k} {ok.mean() * 100:3.0f}% @{np.median(v[ok]):5.1f}" if ok.any() else f"{k}   0%")
    print(f"[gait] {tag} n={n} events (share of episodes, median ref frame): " + " | ".join(parts), flush=True)
    parts = []
    for k in keys_w:
        v = np.array([r.get(k, np.nan) for r in rows], float)
        parts.append(f"{k} {np.nanmedian(v):.3f}" if np.isfinite(v).any() else f"{k} nan")
    print(f"[gait] {tag} window ref {a.window[0]}-{a.window[1]} (medians): " + " | ".join(parts), flush=True)
    # order of events per episode
    from collections import Counter
    orders = Counter()
    for r in rows:
        seq = sorted([(r[k], k) for k in keys_ev if np.isfinite(r[k])])
        orders[" > ".join(k for _, k in seq)] += 1
    for o, c in orders.most_common(4):
        print(f"[gait] {tag} order {c:3d}/{n}: {o}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration")
    p.add_argument("--epoch", type=int, default=3000)
    p.add_argument("--start", default="2,85")
    p.add_argument("--assist", default="0")
    p.add_argument("--assist_from", type=float, default=75)
    p.add_argument("--assist_ramp", type=float, default=10)
    p.add_argument("--seeds", default="101")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--horizon", type=int, default=0, help="0 = to the clip end (or 60 frames from a late start)")
    p.add_argument("--window", default="85,145")
    p.add_argument("--out", default="", help="folder for the per-config traces (.npz) and episode rows (.json)")
    p.add_argument("--_one", default="", help=argparse.SUPPRESS)
    a = p.parse_args()
    a.seeds = [int(x) for x in a.seeds.split(",")]
    a.window = [int(x) for x in a.window.split(",")]
    if a._one:
        s, al = a._one.split(":")
        start, assist = int(s), float(al)
        import numpy as np
        if a.horizon == 0 and start > 2:
            a.horizon = 60
        out, meta = run_one(a, start, assist)
        tag = f"{a.iteration} e{a.epoch} start {start} assist {assist:g}"
        rows = summarize(tag, out, meta, a)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            base = os.path.join(a.out, f"s{start}_a{assist:g}")
            np.savez_compressed(base + ".npz", **{f"{sd}_{k}": v for sd, D in out.items() for k, v in D.items()})
            json.dump(rows, open(base + ".json", "w"), indent=0, default=float)
        return
    for s in a.start.split(","):
        for al in a.assist.split(","):
            cmd = [sys.executable, os.path.abspath(__file__), a.iteration, "--epoch", str(a.epoch), "--seeds",
                   ",".join(map(str, a.seeds)), "--num_envs", str(a.num_envs), "--horizon", str(a.horizon),
                   "--window", ",".join(map(str, a.window)), "--assist_from", str(a.assist_from),
                   "--assist_ramp", str(a.assist_ramp), "--out", a.out, "--_one", f"{s}:{al}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[gait]")]
            print("\n".join(lines) if lines else f"[gait] start {s} assist {al}: failed\n{r.stderr[-2500:]}",
                  flush=True)


if __name__ == "__main__":
    main()
