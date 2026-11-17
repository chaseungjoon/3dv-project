"""STATUS (2026-11-17, fix-2 investigation): NEGATIVE for holding (2-8 mm: 0/42 static holds; with the fix-2 solver
5/24 vs 15-19/24 without it). It did raise grasp forces during the pickup (35-84 N vs <= 7 N) and revealed that the
reference left fingers sit 13-15 mm INSIDE the book while the thumb is 3.6 mm outside. Not used by any iteration.

Contact-consistent reference: close the reference fingers slightly INTO the object where the demonstration touches
it, so that a policy (or a PD controller) that imitates the reference actually squeezes the object. CPU, seconds.

    python environment_iteration/src/tools/contact_reference.py --pen 0.004 --out place_book_fix2 [--force]

Why: mocap fingers sit ON the object surface (zero penetration). A rigid simulated hand driven to exactly that pose
exerts no grip force, so the demonstration cannot be executed: a perfect imitator (root pinned to the reference,
joints PD-driven to it) never lifts the book (environment_iteration/src/probes/grip.py --replay; fix-2 ITERATION.md).
A real fingertip is soft: its bone ends up a few mm closer to the object than the skin contact. This tool moves the
reference finger joints so the distal finger segment (thumb IP, finger DIP capsule) is `--pen` metres inside the
object's collision box in every frame where the reference already has it within `--contact` of the surface.

Per frame and finger: damped least squares on the signed gap (capsule segment to box, minus radius) over the 3 joints
of that finger nearest the tip (thumb CMC/MCP/IP, fingers MCP/PIP/DIP; 9 exp-map values), using the exp-map forward
kinematics of tools/hand_fk.py (0.37 cm from the stored joint positions). The correction is weighted by contact
(1 within `--contact`, fading to 0 at `--contact` + 1 cm) and smoothed over +-2 frames. Stored joint positions
(columns 189:402) of the changed bodies move by the FK difference. Everything else in the file is untouched.
Writes skillmimic/data/motions/ParaHome/<out>/<same file name>. Refuses to overwrite (motions are never edited).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_fk as H  # noqa: E402

BOX = os.path.join(H.REPO, "skillmimic/data/assets/obj/book_fix/box.obj")
MOTIONS = os.path.join(H.REPO, "skillmimic/data/motions/ParaHome")
FINGERS = {"thumb": ["FirstCMC", "FirstMCP", "IP"], "index": ["SecondMCP", "SecondPIP", "SecondDIP"],
           "middle": ["ThirdMCP", "ThirdPIP", "ThirdDIP"], "ring": ["FourthMCP", "FourthPIP", "FourthDIP"],
           "little": ["FifthMCP", "FifthPIP", "FifthDIP"]}


def box_frame():
    c8 = np.array([[float(x) for x in ln.split()[1:4]] for ln in open(BOX) if ln.startswith("v ")])
    c = c8.mean(0)
    ax = [c8[1] - c8[0], c8[2] - c8[0], c8[4] - c8[0]]
    h = np.array([np.linalg.norm(a) / 2 for a in ax])
    Rb = np.stack([a / np.linalg.norm(a) for a in ax], 1)
    return c, Rb, h


class Box:
    def __init__(self):
        self.c, self.Rb, self.h = box_frame()

    def sd(self, pts, obj_pos, obj_R):
        """Signed distance of world points to the object's box at this pose (negative inside)."""
        loc = ((pts - obj_pos) @ obj_R - self.c) @ self.Rb
        d = np.abs(loc) - self.h
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


class Finger:
    def __init__(self, m, hand, name):
        self.bodies = [m.index[f"j{hand}{b}"] for b in FINGERS[name]]
        tip = self.bodies[-1]
        a, b, r = m.capsules[tip][0]
        self.seg = (a, b)
        self.r = r
        self.dofs = [m.dofs[i][0][0] for i in self.bodies]          # first dof of each joint (3 consecutive)
        self.idx = np.concatenate([np.arange(d, d + 3) for d in self.dofs])
        # chain from the wrist's child down to the tip (all ancestors up to the wrist)
        chain, i = [], tip
        while "Wrist" not in m.names[i]:
            chain.append(i); i = m.parent[i]
        self.chain, self.wrist = chain[::-1], i


def chain_fk(m, f, P_w, R_w, dof):
    """Positions / rotations of a finger chain given the wrist pose."""
    P, R = {f.wrist: P_w}, {f.wrist: R_w}
    for i in f.chain:
        p = m.parent[i]
        P[i] = P[p] + R[p] @ m.offset[i]
        d0 = m.dofs[i][0][0]
        R[i] = R[p] @ H.exp_to_mat(dof[d0:d0 + 3])
    return P, R


def gap(m, f, box, P_w, R_w, dof, obj_pos, obj_R):
    P, R = chain_fk(m, f, P_w, R_w, dof)
    t = f.bodies[-1]
    a, b = P[t] + R[t] @ f.seg[0], P[t] + R[t] @ f.seg[1]
    pts = a + np.linspace(0, 1, 7)[:, None] * (b - a)
    return float(box.sd(pts, obj_pos, obj_R).min() - f.r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="place_book", help="motion folder under skillmimic/data/motions/ParaHome")
    p.add_argument("--out", required=True)
    p.add_argument("--pen", type=float, default=0.004, help="target penetration of the distal segment (m)")
    p.add_argument("--contact", type=float, default=0.012, help="reference gap below which a finger is in contact")
    p.add_argument("--max_step", type=float, default=0.6, help="max change of one joint's exp map (rad)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    import torch
    src_dir = os.path.join(MOTIONS, a.src)
    files = [x for x in os.listdir(src_dir) if x.endswith(".pt")]
    assert len(files) == 1, f"one .pt expected in {src_dir}"
    out_dir = os.path.join(MOTIONS, a.out)
    if os.path.exists(out_dir) and not a.force:
        sys.exit(f"{out_dir} exists; motions are never overwritten (pick a new name)")
    raw = torch.load(os.path.join(src_dir, files[0]), map_location="cpu")
    data = raw.cpu().numpy().astype(np.float64)
    T = data.shape[0]
    m = H.Model()
    box = Box()
    fingers = [(hand, nm, Finger(m, hand, nm)) for hand in ("Left", "Right") for nm in FINGERS]
    delta = np.zeros((T, 180))
    weight = np.zeros((T, len(fingers)))
    report = []
    for fr in range(T):
        dof = data[fr, H.DOF].copy()
        Pall, Rall = m.fk(data[fr, H.ROOT_POS], H.exp_to_mat(data[fr, H.ROOT_ROT]), dof)
        op, oR = data[fr, H.OBJ_POS], H.exp_to_mat(data[fr, H.OBJ_ROT])
        for k, (hand, nm, f) in enumerate(fingers):
            Pw, Rw = Pall[f.wrist], Rall[f.wrist]
            g0 = gap(m, f, box, Pw, Rw, dof, op, oR)
            w = float(np.clip((a.contact + 0.01 - g0) / 0.01, 0, 1))
            weight[fr, k] = w
            if w == 0:
                continue
            q = dof.copy()
            target = -a.pen
            for _ in range(25):
                g = gap(m, f, box, Pw, Rw, q, op, oR)
                if abs(g - target) < 3e-4:
                    break
                J = np.zeros(len(f.idx))
                for j, di in enumerate(f.idx):
                    q2 = q.copy(); q2[di] += 1e-3
                    J[j] = (gap(m, f, box, Pw, Rw, q2, op, oR) - g) / 1e-3
                step = J * (target - g) / (J @ J + 1e-4)
                q[f.idx] += np.clip(step, -0.1, 0.1)
                dq = (q[f.idx] - dof[f.idx]).reshape(3, 3)
                n = np.linalg.norm(dq, axis=1, keepdims=True)
                dq = dq * np.minimum(1, a.max_step / np.maximum(n, 1e-9))
                q[f.idx] = dof[f.idx] + dq.ravel()
            delta[fr, f.idx] = w * (q[f.idx] - dof[f.idx])
    # temporal smoothing (+-2 frames) of the corrections
    ker = np.array([1, 2, 3, 2, 1], float); ker /= ker.sum()
    sm = np.stack([np.convolve(np.pad(delta[:, j], 2, mode="edge"), ker, mode="valid") for j in range(180)], 1)
    new = data.copy()
    new[:, H.DOF] = data[:, H.DOF] + sm
    # move the stored joint positions of changed bodies by the FK difference
    for fr in range(T):
        P0, _ = m.fk(data[fr, H.ROOT_POS], H.exp_to_mat(data[fr, H.ROOT_ROT]), data[fr, H.DOF])
        P1, _ = m.fk(new[fr, H.ROOT_POS], H.exp_to_mat(new[fr, H.ROOT_ROT]), new[fr, H.DOF])
        bp = new[fr, H.BODY].reshape(71, 3)
        bp[:61] += P1 - P0
        new[fr, H.BODY] = bp.ravel()
    # report
    for k, (hand, nm, f) in enumerate(fingers):
        frames = np.nonzero(weight[:, k] > 0)[0]
        if not len(frames):
            continue
        g_before, g_after = [], []
        for fr in frames:
            op, oR = data[fr, H.OBJ_POS], H.exp_to_mat(data[fr, H.OBJ_ROT])
            for dat, out in ((data, g_before), (new, g_after)):
                Pall, Rall = m.fk(dat[fr, H.ROOT_POS], H.exp_to_mat(dat[fr, H.ROOT_ROT]), dat[fr, H.DOF])
                out.append(gap(m, f, box, Pall[f.wrist], Rall[f.wrist], dat[fr, H.DOF], op, oR))
        dj = np.abs(sm[frames][:, f.idx]).max()
        report.append(f"  {hand:5s} {nm:6s}: {len(frames):3d} frames ({frames.min()}-{frames.max()}), gap median "
                      f"{np.median(g_before) * 1000:+5.1f} -> {np.median(g_after) * 1000:+5.1f} mm, max joint change "
                      f"{dj:.2f} rad")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.tensor(new, dtype=raw.dtype), os.path.join(out_dir, files[0]))
    with open(os.path.join(out_dir, "README.txt"), "w") as fo:
        fo.write(f"Generated by environment_iteration/src/tools/contact_reference.py --src {a.src} --pen {a.pen} "
                 f"--contact {a.contact} --max_step {a.max_step}\nPer finger (gap = distal capsule to book box, "
                 f"negative = inside):\n" + "\n".join(report) + "\n")
    print(f"[contact_ref] {a.src} -> {a.out}: pen {a.pen * 1000:.1f} mm")
    print("\n".join(report))


if __name__ == "__main__":
    main()
