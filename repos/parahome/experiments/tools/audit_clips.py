"""ParaHome clip audit (CPU only): how hard is each Place-Book / Drink-Cup / Kettle clip, kinematically?

For every annotated interval of the target actions it measures, from the raw mocap (data/seq):
which hand(s) hold the object, how high and how far the object moves, how far the person walks while
holding it, how fast the grasp turns into a lift, and how often hand joints sit inside the object mesh
(capture noise). It also marks the clips that already exist in simulation format in
../t09-wristmimic/InterAct/Parahome (the WristMimic runs of this project use those).

    python experiments/tools/audit_clips.py                 # -> experiments/results/audit/{clips.csv,AUDIT.md}

Distances are hand *joint centres* (ParaHome joint_positions, 25 per hand) to object surface points
(20k samples of data/scan/<obj>/simplified/base.obj). Joint centres sit ~1 cm under the skin, so
"hand near object" uses NEAR_M = 2.5 cm. These thresholds were fixed before looking at any result.
The scans are not watertight and their normals are mixed (cup 56% outward), so only the unsigned
distance is used; hand-object penetration cannot be measured reliably and is not reported.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
T09_INTERACT = os.path.join(REPO, "..", "t09-wristmimic", "InterAct", "Parahome")
FPS = 30.0
NEAR_M = 0.025       # hand joint centre within 2.5 cm of the object surface -> touching
LIFT_M = 0.03        # object counts as lifted 3 cm above its start height
LHAND = list(range(23, 48))
RHAND = list(range(48, 73))

CATEGORIES = [  # (category, object, regex on the lower-cased annotation)
    ("drink_cup", "cup", r"^drink from cup$"),
    ("place_book", "book", r"^move book from (desk|table|bookshelf) to (desk|table|bookshelf)$"),
    ("move_kettle", "kettle", r"^move kettle from (table|desk) to (table|desk)$"),
    # ParaHome has no button. The nearest "operate a control" action is the gas-stove knob (articulated).
    ("turn_gasstove", "gasstove", r"^turn (on|off) gas stove$"),
]


def load_points(obj, n=20000, cache={}):
    if obj not in cache:
        m = trimesh.load(os.path.join(REPO, "data", "scan", obj, "simplified", "base.obj"), force="mesh", process=False)
        pts, face = trimesh.sample.sample_surface(m, n, seed=0)
        normals = m.face_normals[face]
        cache[obj] = (cKDTree(pts), pts, normals, m.bounding_box.extents.copy())
    return cache[obj]


def surface_dist(obj, T, joints):
    """joints (F, J, 3) world -> unsigned distance to the object surface (F, J)."""
    tree, _, _, _ = load_points(obj)
    Rm, t = T[:, :3, :3], T[:, :3, 3]
    local = np.einsum("fij,fkj->fki", Rm.transpose(0, 2, 1), joints - t[:, None])
    d, _ = tree.query(local.reshape(-1, 3))
    return d.reshape(local.shape[:2])


def first(mask):
    i = np.flatnonzero(mask)
    return int(i[0]) if i.size else -1


def audit(seq, a, b, obj, part="base"):
    S = os.path.join(REPO, "data", "seq", seq)
    jp = load_seq(S, "joint_positions")[a:b]
    otf = load_seq(S, "object_transformations")
    key = f"{obj}_{part}"
    if key not in otf[0]:
        return None
    T = np.stack([otf[f][key] for f in range(a, b)]).astype(np.float64)
    n = T.shape[0]
    op = T[:, :3, 3]
    dl = surface_dist(obj, T, jp[:, LHAND]); dr = surface_dist(obj, T, jp[:, RHAND])
    near_l, near_r = dl.min(1) < NEAR_M, dr.min(1) < NEAR_M
    near = near_l | near_r
    lifted = op[:, 2] > op[0, 2] + LIFT_M
    held_lift = near & lifted
    root = jp[:, 0]
    step = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    ostep = np.linalg.norm(np.diff(op, axis=0), axis=1)
    speed = np.convolve(ostep, np.ones(5) / 5, mode="same") * FPS
    g, l = first(near), first(lifted)
    _, _, _, ext = load_points(obj)
    return dict(
        frames=n, seconds=round(n / FPS, 2),
        hand=("both" if near_l.sum() >= 10 and near_r.sum() >= 10 else "left" if near_l.sum() > near_r.sum() else "right"),
        near_frames_l=int(near_l.sum()), near_frames_r=int(near_r.sum()),
        grasp_frame=g, lift_frame=l, grasp_to_lift_frames=(l - g) if g >= 0 and l >= 0 else None,
        lift_m=round(float(op[:, 2].max() - op[0, 2]), 3),
        travel_m=round(float(np.linalg.norm(op[-1] - op[0])), 3),
        path_m=round(float(ostep.sum()), 3), peak_speed=round(float(speed.max()), 2),
        walk_m=round(float(step.sum()), 3),
        carry_walk_m=round(float(step[held_lift[1:]].sum()), 3),
        held_lift_frames=int(held_lift.sum()),
        hand_obj_min_cm=round(float(np.minimum(dl.min(1), dr.min(1))[near].mean() * 100), 2) if near.any() else None,
        obj_size_m="x".join(f"{e:.2f}" for e in sorted(ext, reverse=True)),
    )


_SEQ = {}


def load_seq(S, name):
    k = (S, name)
    if k not in _SEQ:
        if len(_SEQ) > 8:
            _SEQ.clear()
        with open(os.path.join(S, name + ".pkl"), "rb") as f:
            _SEQ[k] = pickle.load(f)
    return _SEQ[k]


def sim_ready_clips():
    """Clips already converted for WristMimic (InterAct format) -> {(seq, a, b): folder}."""
    out = {}
    try:
        import torch  # only for reading the .pt metadata; optional
    except ImportError:
        torch = None
    for d in sorted(glob.glob(os.path.join(T09_INTERACT, "*"))):
        pts = glob.glob(os.path.join(d, "*.pt"))
        if not pts:
            continue
        meta = None
        if torch is not None:
            meta = torch.load(pts[0], map_location="cpu")["metadata"]
        else:
            js = os.path.join(REPO, "experiments", "configs", "interact_meta.json")
            if os.path.isfile(js):
                meta = json.load(open(js)).get(os.path.basename(d))
        if meta:
            a, b = map(int, meta["interval_key"].split())
            out[(meta["scene"], a, b)] = (os.path.basename(d), meta["expression"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/audit")
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    sims = sim_ready_clips()
    rows = []
    seqs = sorted(glob.glob("data/seq/s*"), key=lambda p: int(os.path.basename(p)[1:]))
    for S in seqs:
        seq = os.path.basename(S)
        ann = json.load(open(os.path.join(S, "text_annotation.json")))
        for k, v in ann.items():
            text = v.strip().lower()
            for cat, obj, rx in CATEGORIES:
                if not re.match(rx, text):
                    continue
                s0, s1 = map(int, k.split())
                part = "base" if obj != "gasstove" else "base"
                r = audit(seq, s0, s1, obj, part)
                if r is None:
                    continue
                r.update(category=cat, seq=seq, start=s0, end=s1, text=v.strip(), sim_clip="")
                rows.append(r)
    # InterAct (simulation) intervals are shifted a little from the raw annotation -> audit them as is
    for (seq, s0, s1), (folder, expr) in sims.items():
        text = expr.strip().lower()
        for cat, obj, rx in CATEGORIES:
            if re.match(rx, text):
                r = audit(seq, s0, s1, obj)
                if r:
                    r.update(category=cat, seq=seq, start=s0, end=s1, text=expr, sim_clip=folder)
                    rows.append(r)
    cols = ["category", "seq", "start", "end", "text", "sim_clip", "frames", "seconds", "hand", "near_frames_l",
            "near_frames_r", "grasp_frame", "lift_frame", "grasp_to_lift_frames", "lift_m", "travel_m", "path_m",
            "peak_speed", "walk_m", "carry_walk_m", "held_lift_frames", "hand_obj_min_cm", "obj_size_m"]
    with open(os.path.join(a.out, "clips.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    def stat(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        if v.size == 0:
            return "-"
        return f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]"

    L = ["# ParaHome clip audit", "",
         "자동 생성: `experiments/tools/audit_clips.py` (CPU). 손 관절 중심과 물체 표면의 거리로 계산한 운동학 지표이다.",
         f"손이 물체에 닿음 = 관절 중심이 표면에서 {NEAR_M * 100:.1f} cm 이내, 들어올림 = 시작 높이 + {LIFT_M * 100:.0f} cm 이상, "
         "스캔 mesh가 닫혀 있지 않아 관통은 잴 수 없다 (부호 없는 거리만 사용).", "",
         "## 1. 행동별 요약 (원본 annotation 구간, 중앙값 [25%, 75%])", "",
         "| 행동 | 구간 수 | 길이 (s) | 들어올림 (m) | 물체 이동 (m) | 최고 속도 (m/s) | 들고 걸은 거리 (m) | 잡기→들기 (frame) | 두 손 비율 |",
         "|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        if not r["sim_clip"]:
            by[r["category"]].append(r)
    for cat, _, _ in CATEGORIES:
        rs = by.get(cat, [])
        if not rs:
            continue
        L.append(f"| {cat} | {len(rs)} | {stat(r['seconds'] for r in rs)} | {stat(r['lift_m'] for r in rs)} | "
                 f"{stat(r['travel_m'] for r in rs)} | {stat(r['peak_speed'] for r in rs)} | {stat(r['carry_walk_m'] for r in rs)} | "
                 f"{stat(r['grasp_to_lift_frames'] for r in rs)} | {np.mean([r['hand'] == 'both' for r in rs]):.2f} |")
    L += ["", "## 2. 시뮬레이션 형식으로 준비된 clip (`../t09-wristmimic/InterAct/Parahome`)", "",
          "| clip | 행동 | frame | 손 | 잡기 frame | 들기 frame | 들어올림 (m) | 물체 이동 (m) | 들고 걸은 거리 (m) | 최고 속도 | 잡는 동안 손-물체 거리 (cm) | 물체 크기 (m) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in rows if r["sim_clip"]], key=lambda r: (r["category"], r["sim_clip"])):
        L.append(f"| `{r['sim_clip']}` | {r['category']} | {r['frames']} | {r['hand']} | {r['grasp_frame']} | {r['lift_frame']} | "
                 f"{r['lift_m']} | {r['travel_m']} | {r['carry_walk_m']} | {r['peak_speed']} | {r['hand_obj_min_cm']} | {r['obj_size_m']} |")
    L += ["", "ParaHome에는 버튼이 없다. `turn_gasstove`(가스레인지 손잡이 돌리기, 관절 물체)는 '조작부를 누르거나 돌리는' 가장 가까운 행동으로만 "
          "참고용으로 넣었다. Push Button은 HumanoidMimicGen 쪽에서 다룬다.", ""]
    open(os.path.join(a.out, "AUDIT.md"), "w").write("\n".join(L) + "\n")
    print(f"[audit] {len(rows)} clips ({sum(1 for r in rows if r['sim_clip'])} sim-ready) -> {a.out}/AUDIT.md")


if __name__ == "__main__":
    main()
