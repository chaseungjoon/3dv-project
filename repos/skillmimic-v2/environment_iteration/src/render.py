"""Videos of the failure moments of one iteration (LOOP.md step 3). GPU; ~1-3 min per rollout of 8 envs.

    python environment_iteration/src/render.py <iteration> [--clips iterations/<it>/clips.json] [--epoch E]
    python environment_iteration/src/render.py <iteration> --cell stoch --seed 101 --window 60:110   # ad hoc

Why a fresh rollout: GPU PhysX is not bit-reproducible, so the episodes of the evaluation cannot be replayed
exactly. render.py runs the same policy / cell / seed again with cameras on, re-detects the failure events in
THIS rollout (src/metrics.py) and cuts the clips there. Clips are therefore chosen by failure mode / event / frame
window (clips.json), never by evaluation env index. The numbers in the HUD are this rollout's.

Each frame = two views side by side + HUD:
  left   wide, fixed camera across the desk that sees the whole reference path
  right  close-up following the book
  green wireframe = where the reference book is at this frame; green dots = reference key bodies (head, elbows,
  knees, ankles, fingertips). The red box is the simulated book.
  HUD: frame (t, 0..147) and reference frame, reference phase, book error / height / distance vs reference,
  wrist distances, fingertip contact force, status (IN HAND / FREE / ON FLOOR / FALLEN, TRACK OK / LOST), and a
  timeline with the reference phases and this episode's events (grasp, track lost, drop, floor, fall).

Output: video/<iteration>/e<epoch>_<cell>_s<seed>[_<tag>]/
  full/env<NN>.mp4        whole episode, real time (30 fps)
  clips/<id>__env<NN>__f<a>-<b>.mp4   slow motion (--slowmo, default 0.5x)
  sheets/<id>.png         contact sheet: 5 frames around the anchor for each selected episode
  INDEX.md                what each clip shows (mode, event frames, why it was requested)
  trace.npz, events.json  this rollout's trace and failure events

clips.json (iterations/<it>/clips.json; written by the agent from analysis/, edited by the human):
  {"epoch": 3000,
   "rollouts": [{"cell": "stoch", "seed": 101, "num_envs": 8, "horizon": 180}],
   "clips": [{"id": "lift_lag", "cell": "stoch", "seed": 101,
              "select": {"mode": ["lag", "off_path"]} | {"event": "drop"} | {"outcome": "fall_back"} | {"all": true},
              "anchor": "first_bad" | "grasp" | "off_desk" | "rot_off" | "drop" | "floor" | "fall" | "frame",
              "pre": 20, "post": 15,            (anchor-relative window)  or  "start": 40, "end": 75 (anchor "frame")
              "max_episodes": 3, "why": "what to look for"}]}
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import iterlib as L  # noqa: E402
import metrics as MX  # noqa: E402

KEY_NAMES = ["head", "lknee", "rknee", "lelbow", "relbow", "lankle", "rankle"] + [f"lf{i}" for i in range(5)] + \
            [f"rf{i}" for i in range(5)]
EVENT_COLORS = {"grasp": (60, 180, 60), "off_desk": (120, 160, 40), "rot_off": (180, 60, 180), "track_lost": (0, 140, 255), "drop": (40, 40, 230), "floor": (30, 80, 140),
                "fall": (20, 20, 20), "clip_end": (150, 150, 150)}                     # BGR
PHASE_COLORS = {"static": (200, 200, 200), "lift": (170, 220, 250), "carry": (180, 240, 200),
                "placed": (230, 200, 230), "after_clip": (235, 235, 235)}


quat_rotate = MX.quat_rotate


def book_box(it):
    vs = MX.load_box(L.REPO, it.get("object_box", MX.BOOK_BOX))
    if vs is None:
        return np.zeros((0, 3)), []
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return vs, edges


class Cameras:
    """Two camera sensors per env, created inside the task's _build_env (camera sensors must exist before
    prepare_sim). Positions are env-local (like the root-state tensors); the view matrix works in global
    coordinates (env-local + env origin)."""

    def __init__(self, width=640, height=480):
        self.W, self.H = width, height
        self.handles = {}

    def install(self, task_cls):
        from isaacgym import gymapi
        from env.tasks.base_task import BaseTask
        BaseTask.__init__.__defaults__ = (True,)      # enable_camera_sensors: keep a graphics device when headless
        orig = task_cls._build_env
        cams = self

        def _build_env(self_, env_id, env_ptr, humanoid_asset, _o=orig):
            _o(self_, env_id, env_ptr, humanoid_asset)
            hs = []
            for fov in (70.0, 55.0):
                pr = gymapi.CameraProperties()
                pr.width, pr.height, pr.horizontal_fov = cams.W, cams.H, fov
                hs.append(self_.gym.create_camera_sensor(env_ptr, pr))
            cams.handles[env_id] = (env_ptr, hs)
        task_cls._build_env = _build_env

    def attach(self, env):
        from isaacgym import gymapi
        self.env, self.gym, self.sim, self.gymapi = env, env.task.gym, env.task.sim, gymapi
        self.origin = {}
        for e, (ep, _) in self.handles.items():
            o = self.gym.get_env_origin(ep)
            self.origin[e] = np.array([o.x, o.y, o.z])
        ro, rr = env.ref_obj_path, env.ref_root_path
        pts = np.r_[ro[:, :2], rr[:, :2]]
        c = (pts.min(0) + pts.max(0)) / 2
        n = ro[0, :2] - rr[0, :2]
        n = n / (np.linalg.norm(n) + 1e-9)                   # human -> book: the camera looks across the desk
        span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        self.wide_pos = np.r_[c + n * (0.9 + 0.75 * span), 1.65]
        self.wide_tgt = np.r_[c, 0.85]
        ang = np.radians(40)
        self.close_dir = np.array([n[0] * np.cos(ang) - n[1] * np.sin(ang), n[0] * np.sin(ang) + n[1] * np.cos(ang)])
        self.track = {}
        self.box, self.edges = book_box(env.it)

    def capture(self, vals):
        """Place cameras, render, return {env: (wide_img, close_img, (V, P) per view)} for the current state."""
        gym, sim, ga = self.gym, self.sim, self.gymapi
        obj = vals["obj"].detach().cpu().numpy()
        for e, (ep, (cw, cc)) in self.handles.items():
            gym.set_camera_location(cw, ep, ga.Vec3(*self.wide_pos), ga.Vec3(*self.wide_tgt))
            b = obj[e]
            b = np.array([b[0], b[1], max(b[2], 0.3)])
            tr = self.track.get(e, b)
            tr = 0.8 * tr + 0.2 * b
            self.track[e] = tr
            pos = np.r_[tr[:2] + self.close_dir * 1.25, tr[2] + 0.7]
            gym.set_camera_location(cc, ep, ga.Vec3(*pos), ga.Vec3(tr[0], tr[1], tr[2] - 0.05))
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        out = {}
        for e, (ep, hs) in self.handles.items():
            views = []
            for h in hs:
                img = gym.get_camera_image(sim, ep, h, ga.IMAGE_COLOR).reshape(self.H, self.W, 4)[..., :3]
                V = np.array(gym.get_camera_view_matrix(sim, ep, h))
                P = np.array(gym.get_camera_proj_matrix(sim, ep, h))
                views.append((img[..., ::-1].copy(), V, P))      # RGB -> BGR for cv2
            out[e] = views
        return out

    def project(self, e, pts, V, P):
        p = np.c_[pts + self.origin[e], np.ones(len(pts))] @ V
        z = p[:, 2]
        ok = z < -0.02
        zz = np.where(ok, z, -1.0)
        u = self.W / 2 - p[:, 0] * self.W * P[0, 0] / (2 * zz)
        v = self.H / 2 + p[:, 1] * self.H * P[1, 1] / (2 * zz)
        return np.c_[u, v], ok

    def overlay(self, e, img, V, P, ref_pos, ref_rot, ref_keys):
        import cv2
        if len(self.edges):
            corners = quat_rotate(np.repeat(ref_rot[None], 8, 0), self.box) + ref_pos
            uv, ok = self.project(e, corners, V, P)
            for i, j in self.edges:
                if ok[i] and ok[j]:
                    cv2.line(img, tuple(int(x) for x in uv[i]), tuple(int(x) for x in uv[j]), (60, 220, 60), 2,
                             cv2.LINE_AA)
        uv, ok = self.project(e, ref_keys, V, P)
        for k in range(len(uv)):
            if ok[k]:
                cv2.circle(img, (int(uv[k, 0]), int(uv[k, 1])), 4, (60, 220, 60), -1, cv2.LINE_AA)
        return img


def hud(frame_img, t, e, S, info, ev, c, phases, title):
    """Bottom panel: numbers of this frame + timeline with phases and the episode's events."""
    import cv2
    W = frame_img.shape[1]
    Hh = 136
    panel = np.full((Hh, W, 3), 255, np.uint8)
    nc = c["n_clip"]
    obj, ref = S["obj"][t, e], S["ref_obj"][t, e]
    err = float(np.linalg.norm(obj - ref))
    r0 = info["ref_start"]
    dist, rdist = float(np.linalg.norm(obj[:2] - r0[:2])), float(np.linalg.norm(ref[:2] - r0[:2]))
    al = bool(S["alive"][t, e])
    wd = min(S["wrist_d_r"][t, e], S["wrist_d_l"][t, e])
    in_hand = al and wd < MX.HAND_R and obj[2] - c["obj_z0"] >= MX.IN_HAND_DZ
    rerr = info["rot_err"][e]
    rot_txt = f" | orientation vs ref {rerr[t]:.0f} deg" if rerr is not None else ""
    status = "FALLEN" if not al else "ON FLOOR" if obj[2] < MX.FLOOR_Z else "IN HAND" if in_hand else "FREE"
    track = "TRACK OK" if (err <= MX.TASK_EPS_M and al and t < nc) else "TRACK LOST" if t < nc else "AFTER CLIP"
    ph = MX.phase_of(t, phases) if t < nc else "after_clip"
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, f"{title} | env {e} | frame {t}/{nc - 1} (ref {int(S['ref_frame'][t, e])}) | phase: {ph} | "
                f"first failure: {ev['mode']} @ {ev['first_bad']}", (8, 20), f, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, f"book err {err:.2f} m | height {obj[2]:.2f} (ref {ref[2]:.2f}) | distance {dist:.2f} "
                f"(ref {rdist:.2f}) | wrist R {S['wrist_d_r'][t, e]:.2f} L {S['wrist_d_l'][t, e]:.2f} m | "
                f"finger force R {S['hand_contact_r'][t, e]:.0f} L {S['hand_contact_l'][t, e]:.0f} N{rot_txt}",
                (8, 42), f, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    col = {"IN HAND": (60, 160, 60), "FREE": (0, 140, 255), "ON FLOOR": (30, 80, 140), "FALLEN": (0, 0, 200)}[status]
    cv2.putText(panel, status, (8, 68), f, 0.7, col, 2, cv2.LINE_AA)
    cv2.putText(panel, track, (160, 68), f, 0.7, (60, 160, 60) if track == "TRACK OK" else (0, 0, 200), 2,
                cv2.LINE_AA)
    # timeline
    T = S["obj"].shape[0]
    x0, x1, y0, y1 = 8, W - 8, 84, 104
    sx = lambda fr: int(x0 + (x1 - x0) * fr / max(1, T - 1))   # noqa: E731
    for name, (a, b) in phases.items():
        cv2.rectangle(panel, (sx(a), y0), (sx(b), y1), PHASE_COLORS.get(name, (220, 220, 220)), -1)
        cv2.putText(panel, name, (sx(a) + 2, y1 - 5), f, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    for k, (fr, kind, _) in enumerate(ev["events"]):
        cv2.line(panel, (sx(fr), y0 - 4), (sx(fr), y1 + 4), EVENT_COLORS.get(kind, (0, 0, 0)), 2)
        cv2.putText(panel, f"{kind} {fr}", (sx(fr) - 10, y1 + 12 + 10 * (k % 2)), f, 0.33,
                    EVENT_COLORS.get(kind, (0, 0, 0)), 1, cv2.LINE_AA)
    cv2.line(panel, (sx(t), y0 - 6), (sx(t), y1 + 6), (255, 0, 255), 2)
    return np.vstack([frame_img, panel])


def anchor_frame(ev, anchor):
    if anchor == "first_bad":
        return ev["first_bad"] if ev["mode"] != "success" else None
    if anchor in ("grasp", "drop", "floor", "off_desk", "rot_off"):
        return ev[f"{anchor}_frame"]
    if anchor == "fall":
        return ev["fall_frame"] if ev["fall_frame"] >= 0 else None
    return None


def select(evs, sel):
    if sel.get("all"):
        return list(evs)
    out = list(evs)
    if "mode" in sel:
        ms = sel["mode"] if isinstance(sel["mode"], list) else [sel["mode"]]
        out = [v for v in out if v["mode"] in ms]
    if "event" in sel:
        out = [v for v in out if anchor_frame(v, sel["event"]) is not None]
    if "outcome" in sel:     # metrics.body_outcome label (walk_up, stand_squat, fall_fwd, fall_back, ...)
        os_ = sel["outcome"] if isinstance(sel["outcome"], list) else [sel["outcome"]]
        out = [v for v in out if v.get("outcome") in os_]
    return out


class Writer:
    def __init__(self, path, size, fps):
        import imageio_ffmpeg
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.g = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                             macro_block_size=8, quality=7)
        self.g.send(None)

    def add(self, bgr):
        self.g.send(np.ascontiguousarray(bgr[..., ::-1]))

    def close(self):
        self.g.close()


def run_group(env, cams, ck, epoch, cell, seed, perturb, clips, out_dir, slowmo, keep_frames):
    import cv2
    tmp = os.path.join(out_dir, ".frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    nk = len(env.key_ids)

    def on_frame(t, vals):
        imgs = cams.capture(vals)
        rp = vals["ref_obj"].detach().cpu().numpy()
        rr = vals["ref_obj_rot"].detach().cpu().numpy()
        rk = vals["ref_key_pos"].detach().cpu().numpy().reshape(env.num_envs, nk, 3)
        for e, views in imgs.items():
            tiles = [cams.overlay(e, img, V, P, rp[e], rr[e], rk[e]) for img, V, P in views]
            cv2.imwrite(os.path.join(tmp, f"e{e:02d}_{t:04d}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])

    t0 = time.time()
    S, info = env.rollout(ck, seed, perturb, on_frame=on_frame, progress=True)
    print(f"[render] rollout + capture {time.time() - t0:.0f}s", flush=True)
    c = dict(n_clip=env.n_clip, obj_z0=env.obj_z0, ref_lift=env.ref_lift, thr=env.hold_thr, paper_z=env.paper_z,
             paper_need=env.paper_need, box=cams.box if len(cams.box) == 8 else None)
    ph = MX.ref_phases(MX.clean_ref(S, c["n_clip"]), c["obj_z0"])
    evs = [MX.failure_events(S, S["fall_t"], c, e, ph) for e in range(env.num_envs)]
    for e, ev in enumerate(evs):
        ev["outcome"], ev["walk_progress_m"], ev["feet_progress_m"] = MX.body_outcome(S, e, c["n_clip"])
    mets = [MX.episode_metrics(S, S["fall_t"], c, e) for e in range(env.num_envs)]
    np.savez_compressed(os.path.join(out_dir, "trace.npz"), **S)
    L.write_json(os.path.join(out_dir, "events.json"),
                 dict(meta=dict(env.meta(), **info, cell=cell, seed=seed, perturb=perturb, ref_phases=ph),
                      episodes=[dict(ev, **m) for ev, m in zip(evs, mets)]))
    hinfo = dict(ref_start=MX.clean_ref(S, c["n_clip"])[0],
                 rot_err=[MX.rot_err_deg(S, e) for e in range(env.num_envs)])
    title = f"{env.name} e{epoch} {cell} s{seed}"
    T = S["obj"].shape[0]

    def frame(e, t):
        img = cv2.imread(os.path.join(tmp, f"e{e:02d}_{t:04d}.jpg"))
        return hud(img, t, e, S, hinfo, evs[e], c, ph, title)

    size = None
    for e in range(env.num_envs):   # full episodes
        w = None
        for t in range(T):
            fr = frame(e, t)
            if w is None:
                size = (fr.shape[1], fr.shape[0])
                w = Writer(os.path.join(out_dir, "full", f"env{e:02d}.mp4"), size, 30)
            w.add(fr)
        w.close()
    rows = []
    for cl in clips:
        chosen = select(evs, cl.get("select", {"all": True}))
        anchor = cl.get("anchor", "frame")
        made = 0
        sheet = []
        for ev in chosen:
            if made >= int(cl.get("max_episodes", 3)):
                break
            if anchor == "frame":
                a, b, af = int(cl["start"]), int(cl["end"]), None
            else:
                af = anchor_frame(ev, anchor)
                if af is None:
                    continue
                a, b = af - int(cl.get("pre", 20)), af + int(cl.get("post", 20))
            a, b = max(0, a), min(T - 1, b)
            e = ev["env"]
            name = f"{cl['id']}__env{e:02d}__f{a:03d}-{b:03d}.mp4"
            w = Writer(os.path.join(out_dir, "clips", name), size, max(1, round(30 * slowmo)))
            for t in range(a, b + 1):
                w.add(frame(e, t))
            w.close()
            mid = af if af is not None else (a + b) // 2
            picks = [min(b, max(a, mid + d)) for d in (-10, -5, 0, 5, 10)]
            sheet.append(np.hstack([cv2.resize(frame(e, t), None, fx=0.4, fy=0.4) for t in picks]))
            rows.append(dict(id=cl["id"], file=f"clips/{name}", env=e, mode=ev["mode"], first_bad=ev["first_bad"],
                             grasp=ev["grasp_frame"], drop=ev["drop_frame"], floor=ev["floor_frame"], rot_off=ev["rot_off_frame"],
                             fall=ev["fall_frame"], window=f"{a}-{b}", anchor=f"{anchor}@{af}" if af is not None
                             else f"frames {a}-{b}", detail=ev["detail"], why=cl.get("why", "")))
            made += 1
        if sheet:
            os.makedirs(os.path.join(out_dir, "sheets"), exist_ok=True)
            wmax = max(s.shape[1] for s in sheet)
            sheet = [np.pad(s, ((0, 0), (0, wmax - s.shape[1]), (0, 0)), constant_values=255) for s in sheet]
            cv2.imwrite(os.path.join(out_dir, "sheets", f"{cl['id']}.png"), np.vstack(sheet))
        if made == 0:
            rows.append(dict(id=cl["id"], file="(none)", env="-", mode="-", first_bad="-", grasp="-", drop="-",
                             floor="-", fall="-", window="-", anchor=anchor, detail="no episode of this rollout "
                             "matched; try more envs (--num_envs) or another seed", why=cl.get("why", "")))
    if not keep_frames:
        shutil.rmtree(tmp, ignore_errors=True)
    return evs, mets, rows


def write_index(out_dir, head, evs, mets, rows):
    lines = [f"# {head}", "", "Fresh rollout with cameras (src/render.py); events re-detected in this rollout.",
             "Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.", "",
             "## Episodes of this rollout", "", "| env | mode | broke at | detail | grasp | drop | floor | fall | "
             "task | lift | walk outcome (root / feet along walk) | full video |",
             "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for ev, m in zip(evs, mets):
        lines.append(f"| {ev['env']} | {ev['mode']} | {ev['first_bad']} | {ev['detail']} | {ev['grasp_frame']} | "
                     f"{ev['drop_frame']} | {ev['floor_frame']} | {ev['fall_frame']} | {m['task_success']} | "
                     f"{m['lift_success']} | {ev.get('outcome', '-')} ({ev.get('walk_progress_m', 0):.2f} / "
                     f"{ev.get('feet_progress_m', 0):.2f} m) | full/env{ev['env']:02d}.mp4 |")
    lines += ["", "## Clips", "", "| clip | file | env | mode | anchor | window | detail | what to look for |",
              "|---|---|---:|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['file']} | {r['env']} | {r['mode']} | {r['anchor']} | {r['window']} | "
                     f"{r['detail']} | {r['why']} |")
    with open(os.path.join(out_dir, "INDEX.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration")
    p.add_argument("--clips", default="", help="clips.json (default iterations/<it>/clips.json)")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--seed_run", type=int, default=None)
    p.add_argument("--cell", default="", help="ad hoc: render this protocol cell (ignores clips.json)")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--window", default="", help="ad hoc: frames a:b clipped for every env")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=180)
    p.add_argument("--slowmo", type=float, default=0.5)
    p.add_argument("--size", default="640x480", help="per view; the video is two views wide")
    p.add_argument("--tag", default="")
    p.add_argument("--keep_frames", action="store_true")
    a = p.parse_args()
    pr = L.protocol()
    P = L.paths(a.iteration, a.seed_run)
    cells = {c["cell"]: c for c in pr["eval"]["final"]}
    if a.cell:
        if a.cell not in cells:
            sys.exit(f"unknown cell {a.cell} (protocol: {list(cells)})")
        win = a.window.split(":") if a.window else ["0", str(a.horizon - 1)]
        spec = dict(epoch=a.epoch, rollouts=[dict(cell=a.cell, seed=a.seed, num_envs=a.num_envs, horizon=a.horizon)],
                    clips=[dict(id="window", cell=a.cell, seed=a.seed, select={"all": True}, anchor="frame",
                                start=int(win[0]), end=int(win[1]), max_episodes=a.num_envs, why="ad hoc window")])
    else:
        cp = a.clips or P["clips"]
        spec = L.read_json(cp)
        if not spec or not spec.get("clips"):
            sys.exit(f"no clips in {L.rel(cp)} (write it from analysis/, see LOOP.md 4, or use --cell/--window)")
    epoch = a.epoch or spec.get("epoch") or pr["train"]["epochs"]
    ck = L.checkpoint_at(P["run_dir"], epoch)
    if ck is None:
        sys.exit(f"no checkpoint for epoch {epoch} in {L.rel(P['run_dir'])}/nn")
    groups = spec.get("rollouts") or [dict(cell=cl.get("cell", "stoch"), seed=cl.get("seed", 101))
                                      for cl in spec["clips"]]
    groups = list({(g["cell"], g["seed"]): g for g in groups}.values())
    n_envs = max(int(g.get("num_envs", a.num_envs)) for g in groups)
    horizon = max(int(g.get("horizon", a.horizon)) for g in groups)
    modes = {cells[g["cell"]]["mode"] for g in groups}
    if len(modes) != 1:
        sys.exit("all rollouts of one render call must share mode (det/stoch); split clips.json")
    W, H = (int(x) for x in a.size.split("x"))
    import rollout as RO
    cams = Cameras(W, H)
    env = RO.Env(a.iteration, n_envs, modes.pop(), horizon, pr["eval"]["start_frame"], seed=groups[0]["seed"],
                 cameras=cams, env_spacing=20.0, first_checkpoint=ck)
    for g in groups:
        cell, seed = g["cell"], int(g["seed"])
        out = os.path.join(P["video_dir"], f"e{epoch}_{cell}_s{seed}" + (f"_{a.tag}" if a.tag else ""))
        os.makedirs(out, exist_ok=True)
        clips = [cl for cl in spec["clips"] if cl.get("cell", cell) == cell and int(cl.get("seed", seed)) == seed]
        print(f"[render] {a.iteration} e{epoch} {cell} s{seed}: {n_envs} envs, {horizon} frames, "
              f"{len(clips)} clip spec(s) -> {L.rel(out)}", flush=True)
        evs, mets, rows = run_group(env, cams, ck, epoch, cell, seed, bool(cells[cell]["perturb"]), clips, out,
                                    a.slowmo, a.keep_frames)
        write_index(out, f"{a.iteration} epoch {epoch}, cell {cell}, seed {seed}", evs, mets, rows)
        n = sum(1 for r in rows if r["file"] != "(none)")
        print(f"[render] {n} clip(s), {n_envs} full episodes -> {L.rel(out)}/INDEX.md", flush=True)


if __name__ == "__main__":
    main()
