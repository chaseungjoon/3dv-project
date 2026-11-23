"""Per-episode metrics and failure events from a rollout trace. Numpy only (no torch / isaacgym).

Frame convention (used everywhere in this loop, same as all earlier result documents):
    frame t = control step t of an evaluation episode, t = 0 .. n_clip-1, after a reset at reference frame
    `start_frame` (2). n_clip = clip_frames - start_frame = 148 for place_book. Arrays are indexed [t, env, ...].

Trace keys (rollout.py writes them; experiments/tools/evaluate.py .npz files have the first six):
    obj (T,E,3)  ref_obj (T,E,3)  root (T,E,3)  wrist_d (T,E)  body_err (T,E)  alive (T,E)
    optional: fall_t (E,)  obj_rot/ref_obj_rot (T,E,4 xyzw)  wrist_r/wrist_l (T,E,3)  wrist_d_r/wrist_d_l (T,E)
              obj_contact (T,E)  hand_contact_r/_l (T,E)  reward (T,E)  ref_frame (T,E)

episode_metrics() reproduces experiments/tools/evaluate.py line by line (src/tests/test_metrics.py checks this
against stored evaluations), so numbers from this loop are comparable with every earlier result.
failure_events() adds the loop's own failure analysis (LOOP.md 4).
"""
import os

import numpy as np

TASK_EPS_M = 0.10
HELD_DZ = 0.05          # evaluate.py "held": object > 5 cm above its start height while a wrist is near it
FLOOR_Z = 0.15          # object centre below this = on the floor (book is 4.7 cm thick; desk top is 0.716 m)
DROP_FALL_M = 0.10      # after leaving the hand the object sinks at least this much ...
DROP_WINDOW = 20        # ... within this many frames
MIN_HOLD = 3            # a grasp streak shorter than this is contact noise, not a hold
HAND_R = 0.30           # failure analysis only: wrist-to-object-centre radius of a hold. evaluate.py uses 0.2, which a
                        # two-hand grip on the 27 cm book often exceeds (measured 0.20-0.46 m while lifting, fix-1)
IN_HAND_DZ = 0.02       # failure analysis only: object at least 2 cm above its start = off the desk
LAG_FRAMES = 4          # nearest reference frame this far behind / ahead = lag / lead
BOOK_BOX = "skillmimic/data/assets/obj/book_fix/box.obj"   # 8 corners of the book collision box (fix-1 book)

MODES = ["success", "fall", "no_reach", "no_lift", "knock", "drop", "lag", "lead", "off_path"]
MODE_HELP = {
    "success": "object within 10 cm of the reference for the whole clip, no fall",
    "fall": "the character fell (pelvis below termination height) while the object was still on track",
    "no_reach": "no wrist came within the hold radius of the object before the object left the track",
    "no_lift": "hands reached the object but it stayed at its start pose while the reference lifted it",
    "knock": "the object moved (pushed, slid, tipped over) without ever being in hand",
    "drop": "the object was held, left the hands, and then left the track",
    "lag": "held, but behind the reference along its path (too slow)",
    "lead": "held, but ahead of the reference along its path (too fast)",
    "off_path": "held, on time, but displaced from the reference path (wrong place / height)",
}


def consts_from(meta, ref_obj_full=None):
    """Constants evaluate.py derives from the motion; meta = an evaluate.py / rollout.py result['meta']."""
    return dict(n_clip=int(meta["clip_frames"]) - int(meta["start_frame"]), obj_z0=float(meta["obj_z0"]),
                ref_lift=float(meta["ref_lift"]), thr=float(meta.get("hold_thr", 0.2)),
                paper_z=float(meta.get("paper_z", 0.9)), paper_need=int(meta.get("paper_need", 60)))


def episode_metrics(S, fall_t, c, e):
    """Metrics of env e, identical to experiments/tools/evaluate.py (keep in sync; tests/test_metrics.py)."""
    T = S["obj"].shape[0]
    nc = min(c["n_clip"], T)
    al = S["alive"][:, e].astype(bool)
    fell = bool(fall_t[e] >= 0)
    ft = int(fall_t[e]) if fell else -1
    fell_in_clip = fell and ft < nc
    err = np.linalg.norm(S["obj"][:nc, e] - S["ref_obj"][:nc, e], axis=-1)
    err_alive = err[al[:nc]] if al[:nc].any() else err[:1]
    bad = (err > TASK_EPS_M) | ~al[:nc]
    first_bad = int(np.argmax(bad)) if bad.any() else nc
    z = S["obj"][:, e, 2]
    near = S["wrist_d"][:, e] < c["thr"]
    held = al & near & (z - c["obj_z0"] > HELD_DZ)
    at_paper = al & (S["root"][:, e, 2] > 0.5) & near & (z > c["paper_z"])
    lift_ok = bool((al & near & (z - c["obj_z0"] >= 0.5 * c["ref_lift"])).any())
    return dict(
        fell=int(fell), fall_frame=ft, fell_in_clip=int(fell_in_clip),
        task_success=int((not fell_in_clip) and float(err.max()) <= TASK_EPS_M),
        task_progress=first_bad / nc,
        obj_err_mean=float(err_alive.mean()), obj_err_max=float(err.max()),
        obj_err_end=float(err[nc - 1]) if al[nc - 1] else None,
        body_err_mean=float(S["body_err"][:nc, e][al[:nc]].mean()) if al[:nc].any() else None,
        lift_max=float((z[al] - c["obj_z0"]).max()) if al.any() else 0.0,
        lift_success=int(lift_ok),
        held_frames=int(held.sum()), held_frames_clip=int(held[:nc].sum()),
        paper_frames=int(at_paper.sum()), paper_success=int(at_paper.sum() > c["paper_need"]),
        paper_success_clip=int(at_paper[:nc].sum() > c["paper_need"]),
    )


def aggregate(eps):
    agg = {k: float(np.mean([e[k] for e in eps if e[k] is not None]))
           for k in eps[0] if any(e[k] is not None for e in eps) and not isinstance(eps[0][k], (str, list, dict))}
    agg["n_episodes"] = len(eps)
    return agg


def fall_t_from_alive(alive):
    """Fall frame per env from the sticky alive mask (first False), -1 if always alive. Only valid when the
    episode cannot end any other way inside the horizon (true for evaluation: fall-only termination)."""
    al = alive.astype(bool)
    out = np.full(al.shape[1], -1, dtype=np.int64)
    for e in range(al.shape[1]):
        dead = np.nonzero(~al[:, e])[0]
        if dead.size:
            out[e] = int(dead[0])
    return out


def clean_ref(S, n_clip):
    """Reference object path (n_clip, 3) of a clean-start env: the env whose first reference point is closest to the
    median over envs (a noisy-start env, evaluate.py only, follows a different reference frame)."""
    r0 = S["ref_obj"][0]
    e = int(np.argmin(np.linalg.norm(r0 - np.median(r0, axis=0), axis=-1)))
    return S["ref_obj"][:n_clip, e]


def ref_phases(ref_obj, obj_z0):
    """Phase boundaries of the reference object path (one env's ref_obj, (nc,3)), data-driven:
    static (object at rest) -> lift (rising to the highest point) -> carry (moving) -> placed (at rest again)."""
    nc = len(ref_obj)
    disp = np.linalg.norm(ref_obj - ref_obj[0], axis=-1)
    moved = np.nonzero(disp > 0.02)[0]
    t_move = int(moved[0]) if moved.size else nc
    t_peak = int(np.argmax(ref_obj[:, 2])) if t_move < nc else nc
    speed = np.r_[np.linalg.norm(np.diff(ref_obj, axis=0), axis=-1) * 30.0, 0.0]   # m/s at 30 Hz
    moving = np.nonzero(speed > 0.05)[0]
    t_stop = int(moving[-1]) + 1 if moving.size else nc
    t_peak = max(t_peak, t_move)
    t_stop = max(t_stop, t_peak)
    return dict(static=(0, t_move), lift=(t_move, t_peak), carry=(t_peak, t_stop), placed=(t_stop, nc))


def phase_of(t, phases):
    for name, (a, b) in phases.items():
        if a <= t < b:
            return name
    return "after_clip"


def _streaks(mask):
    """[(start, end_inclusive)] of True runs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def quat_rotate(q, v):
    """q (..., 4) xyzw, v (..., 3) -> rotated v."""
    qv, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def load_box(repo, rel=BOOK_BOX):
    """Collision box corners (8, 3) of the object in its local frame, None if the file is not an 8-vertex box."""
    path = os.path.join(repo, rel)
    if not os.path.isfile(path):
        return None
    vs = np.array([[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")])
    return vs if len(vs) == 8 else None


def lowest_corner_rise(S, e, c):
    """Height gain of the object's LOWEST corner over its start (needs obj_rot + c['box']), else of its centre.
    The demonstration pivots the book up on one edge (frames ~50-62: tilt 7 -> 47 deg, lowest corner still on the
    desk), so this is NOT the hold signal; it only tells when the book is completely off the desk."""
    box = c.get("box")
    if box is not None and "obj_rot" in S:
        corners = quat_rotate(S["obj_rot"][:, e, None, :], box[None]) + S["obj"][:, e, None, :]
        low = corners[..., 2].min(-1)
        return low - low[0]
    return S["obj"][:, e, 2] - c["obj_z0"]


def rot_err_deg(S, e):
    """Angle (deg) between the object's orientation and the reference's, per frame; None without obj_rot.
    The place_book reference rotates the book by ~180 deg between frames 50 and 90 (flat -> upright + turned)."""
    if "obj_rot" not in S or "ref_obj_rot" not in S:
        return None
    d = np.clip(np.abs((S["obj_rot"][:, e] * S["ref_obj_rot"][:, e]).sum(-1)), 0, 1)
    return np.degrees(2 * np.arccos(d))


WALK_FROM, WALK_TO = 82, 142   # frames t of reference frames 85 / 145: the reference carry walk
WALK_PROGRESS_M = 0.30         # root moved this far along the reference walk direction = it walked (or lunged)
BACK_M = 0.20                  # root moved this far against the walk direction = it went backwards
ANKLES = (5, 6)                # jLeftAnkle / jRightAnkle in the keyBodies list (env.yaml) = key_pos columns


def body_outcome(S, e, n_clip=148):
    """What the body does in the reference carry walk (added in fix-5; not part of the failure modes).

    Walk direction = the reference ankles' mean displacement from frame t 82 to 142 (reference frames 85-145).
    Root progress along it is measured from t 82 up to the fall (or the clip end).
      walk_up      no fall in the clip, root progress >= 0.30 m at some frame
      stand_squat  no fall in the clip, less progress (fix-4's squat in place)
      fall_fwd     fell in the clip after root progress >= 0.30 m (lunge / topple along the walk)
      fall_back    fell in the clip after the root moved >= 0.20 m against the walk direction
      fall_inplace fell in the clip otherwise; fell_pre_walk: fell before t 82
      walk_not_observed  the rollout ended before the reference walk direction could be measured
    Returns (label, max progress m, feet progress m = mean ankle displacement along the walk)."""
    n_clip = min(n_clip, len(S["ref_key_pos"]))
    ft = int(S["fall_t"][e]) if "fall_t" in S else -1
    fell = 0 <= ft < n_clip
    if fell and ft <= WALK_FROM:
        return "fell_pre_walk", 0.0, 0.0
    if n_clip <= WALK_TO:
        return "walk_not_observed", 0.0, 0.0
    end = ft if fell else n_clip - 1
    ra = S["ref_key_pos"][:, e][:, list(ANKLES), :2].mean(1)
    wd = ra[WALK_TO] - ra[WALK_FROM]
    wd = wd / max(np.linalg.norm(wd), 1e-9)
    prog = (S["root"][WALK_FROM:end + 1, e, :2] - S["root"][WALK_FROM, e, :2]) @ wd
    kp = S["key_pos"][:, e][:, list(ANKLES), :2]
    feet = float(((kp[end] - kp[WALK_FROM]) @ wd).mean())
    if not fell:
        lab = "walk_up" if prog.max() >= WALK_PROGRESS_M else "stand_squat"
    elif prog.max() >= WALK_PROGRESS_M:
        lab = "fall_fwd"
    elif prog.min() <= -BACK_M:
        lab = "fall_back"
    else:
        lab = "fall_inplace"
    return lab, float(prog.max()), feet


def failure_events(S, fall_t, c, e, phases=None):
    """Failure analysis of env e (LOOP.md 4). Returns the first-failure mode + frames of every notable event.

    first_bad    first frame the task breaks: object > 10 cm from its reference, or the character has fallen
    mode         why it broke at first_bad (MODES / MODE_HELP), 'success' if it never broke inside the clip
    grasp_frame  first frame of the first hold streak (in hand = alive, a wrist within HAND_R, object centre >= 2 cm
                 above its start; a pivot grip with one edge on the desk counts, as in the demonstration. Looser
                 than evaluate.py's `held`, which stays the metric)
    drop_frame   last in-hand frame before the object fell away (None if never dropped)
    off_desk_frame  first frame the object's lowest corner is >= 2 cm above its start (completely lifted)
    rot_off_frame   first clip frame the object is > 45 deg from the reference orientation (alive)
    floor_frame  first frame the object centre is below 0.15 m
    fall_frame   first frame of the fall (-1 = never fell within the horizon)
    events       [(frame, kind, text)] for the video HUD and clip cutting; kinds: grasp, off_desk, rot_off, drop,
                 floor, fall, track_lost (= first_bad), clip_end
    """
    T = S["obj"].shape[0]
    nc = min(c["n_clip"], T)
    al = S["alive"][:, e].astype(bool)
    obj, ref = S["obj"][:, e], S["ref_obj"][:, e]
    z = obj[:, 2]
    err = np.linalg.norm(obj[:nc] - ref[:nc], axis=-1)
    bad = (err > TASK_EPS_M) | ~al[:nc]
    first_bad = int(np.argmax(bad)) if bad.any() else nc
    wd = S["wrist_d"][:, e]
    near = wd < HAND_R
    lz = lowest_corner_rise(S, e, c)
    held = al & near & (z - c["obj_z0"] >= IN_HAND_DZ)
    rerr = rot_err_deg(S, e)
    phases = phases or ref_phases(ref[:nc], c["obj_z0"])
    ft = int(fall_t[e])

    holds = [(a, b) for a, b in _streaks(held[:T]) if b - a + 1 >= MIN_HOLD]
    grasp = holds[0][0] if holds else None
    drop, floor, off_desk, rot_off = None, None, None, None
    for a, b in holds:   # a hold that ends with the object sinking away from the hands
        if b + 1 >= T or not al[b]:
            continue
        w = slice(b + 1, min(T, b + 1 + DROP_WINDOW))
        lost = ~near[w]
        sank = z[b] - z[w].min() >= DROP_FALL_M
        if lost.any() and sank:
            drop = b
            break
    on_floor = np.nonzero(z < FLOOR_Z)[0]
    floor = int(on_floor[0]) if on_floor.size else None
    od = np.nonzero(al & (lz >= IN_HAND_DZ))[0]
    off_desk = int(od[0]) if od.size else None
    if rerr is not None:
        ro = np.nonzero(al[:nc] & (rerr[:nc] > 45))[0]
        rot_off = int(ro[0]) if ro.size else None

    if first_bad >= nc:
        mode, detail = "success", ""
    elif not al[first_bad] or (0 <= ft <= first_bad):
        mode, detail = "fall", f"fell at frame {ft}"
    else:
        held_before = [h for h in holds if h[0] <= first_bad]
        tl = f", {rerr[first_bad]:.0f} deg from reference orientation" if rerr is not None else ""
        if not held_before:
            disp = float(np.linalg.norm(obj[first_bad, :2] - obj[0, :2]))
            dz = float(z[first_bad] - c["obj_z0"])
            if wd[:first_bad + 1].min() >= HAND_R:
                mode, detail = "no_reach", f"closest wrist {wd[:first_bad + 1].min():.2f} m"
            elif disp < 0.05 and dz < IN_HAND_DZ:
                mode, detail = "no_lift", f"object moved {disp * 100:.0f} cm, rose {dz * 100:.0f} cm{tl}"
            else:
                mode, detail = "knock", f"object moved {disp * 100:.0f} cm, rose {dz * 100:.0f} cm{tl}, never in hand"
        else:
            in_hand = held[max(0, first_bad - 2):first_bad + 1].any()
            if not in_hand:
                mode, detail = "drop", f"last in hand at frame {max(h[1] for h in held_before)}{tl}"
            else:   # still in hand: where along the reference path is it?
                k = int(np.argmin(np.linalg.norm(ref[:min(nc, first_bad + 30)] - obj[first_bad], axis=-1)))
                gap = float(np.linalg.norm(ref[k] - obj[first_bad]))
                rose = f"centre {100 * (z[first_bad] - c['obj_z0']):.0f} cm up vs reference " \
                       f"{100 * (ref[first_bad, 2] - c['obj_z0']):.0f} cm"
                if k < first_bad - LAG_FRAMES:
                    mode, detail = "lag", f"nearest reference frame {k} ({first_bad - k} behind, {gap * 100:.0f} cm); {rose}"
                elif k > first_bad + LAG_FRAMES:
                    mode, detail = "lead", f"nearest reference frame {k} ({k - first_bad} ahead, {gap * 100:.0f} cm); {rose}"
                else:
                    mode, detail = "off_path", f"on time but {gap * 100:.0f} cm off the path; {rose}{tl}"
                if mode in ("lag", "lead"):
                    detail += tl

    events = []
    if grasp is not None:
        events.append((grasp, "grasp", "hold starts"))
    if off_desk is not None:
        events.append((off_desk, "off_desk", "whole object off the desk"))
    if rot_off is not None:
        events.append((rot_off, "rot_off", "> 45 deg from the reference orientation"))
    if drop is not None:
        events.append((drop, "drop", "object leaves the hands"))
    if floor is not None:
        events.append((floor, "floor", "object on the floor"))
    if ft >= 0:
        events.append((ft, "fall", "character fell"))
    if first_bad < nc:
        events.append((first_bad, "track_lost", f"object > 10 cm off ({mode})"))
    events.append((nc - 1, "clip_end", "last reference frame"))
    events.sort()
    return dict(env=e, mode=mode, detail=detail, first_bad=first_bad, phase=phase_of(first_bad, phases),
                grasp_frame=grasp, drop_frame=drop, off_desk_frame=off_desk, rot_off_frame=rot_off, floor_frame=floor, fall_frame=ft,
                hold_streaks=[list(map(int, h)) for h in holds],
                max_err_clip=float(err.max()), err_at_first_bad=float(err[min(first_bad, nc - 1)]),
                events=[(int(f), k, s) for f, k, s in events])
