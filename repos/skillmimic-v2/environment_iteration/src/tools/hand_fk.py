"""Forward kinematics of the SkillMimic ParaHome humanoid MJCF, numpy only (used by tools/contact_reference.py).

    python environment_iteration/src/tools/hand_fk.py       # self-check against the motion file's joint positions

The MJCF is a plain tree: every body has a position offset in its parent frame (no rotated frames) and three hinge
joints (x, y, z) at its origin; the root is a free joint. The motion file stores the root position, the root rotation
(exp map), 180 joint values in MJCF joint order (= Isaac Gym DOF order) and joint positions of 71 bodies whose first
61 are the MJCF bodies in the same order. The 3 values of a joint are an EXP MAP of the joint rotation, not three
sequential hinge angles: exp-map FK reproduces the stored joint positions to 0.5 cm (median), sequential x-y-z
hinges are 14-18 cm off. (Isaac Gym's importer turns the 3 hinges into one spherical joint; the probe measured the
simulated fingertips within 1-2 cm of the reference.)
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ASSET = os.path.join(REPO, "skillmimic/data/assets/mjcf/mocap_parahome_boxhand_histphase.xml")
MOTION = os.path.join(REPO, "skillmimic/data/motions/ParaHome/place_book/002_s6_place_book_2460_2610.pt")

# column layout of a ParaHome motion .pt (skillmimic/utils/paramotion_data_handler.py)
ROOT_POS, ROOT_ROT, DOF, BODY, OBJ_POS, OBJ_ROT, CONTACT = (slice(0, 3), slice(3, 6), slice(9, 189), slice(189, 402),
                                                            slice(402, 405), slice(405, 408), slice(408, 409))


def exp_to_mat(e):
    """(..., 3) exp map -> (..., 3, 3)."""
    e = np.asarray(e, dtype=np.float64)
    th = np.linalg.norm(e, axis=-1, keepdims=True)
    k = np.where(th > 1e-9, e / np.maximum(th, 1e-12), np.array([1.0, 0, 0]))
    K = np.zeros(e.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_exp(R):
    """(3,3) -> exp map (3,)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return v * th


def axis_rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Model:
    def __init__(self, path=ASSET):
        root = ET.parse(path).getroot()
        self.names, self.parent, self.offset, self.dofs, self.capsules = [], [], [], [], []
        dof_i = 0

        def walk(el, par):
            nonlocal dof_i
            for b in el.findall("body"):
                i = len(self.names)
                self.names.append(b.get("name"))
                self.parent.append(par)
                self.offset.append(np.array([float(x) for x in b.get("pos", "0 0 0").split()]))
                d = []
                for j in b.findall("joint"):
                    if j.get("type") == "hinge":
                        ax = int(np.argmax(np.abs([float(x) for x in j.get("axis").split()])))
                        d.append((dof_i, ax)); dof_i += 1
                self.dofs.append(d)
                caps = []
                for g in b.findall("geom"):
                    if g.get("type") == "capsule" and g.get("fromto"):
                        ft = np.array([float(x) for x in g.get("fromto").split()])
                        caps.append((ft[:3], ft[3:], float(g.get("size").split()[0])))
                self.capsules.append(caps)
                walk(b, i)
        walk(root.find("worldbody"), -1)
        self.n_dof = dof_i
        self.index = {n: i for i, n in enumerate(self.names)}

    def fk(self, root_pos, root_R, dof):
        """-> world positions (B,3) and rotations (B,3,3) of all bodies."""
        n = len(self.names)
        P, R = np.zeros((n, 3)), np.zeros((n, 3, 3))
        for i in range(n):
            if self.parent[i] < 0:
                P[i], Rb = root_pos, root_R
            else:
                p = self.parent[i]
                P[i] = P[p] + R[p] @ self.offset[i]
                Rb = R[p]
            if self.dofs[i]:
                d0 = self.dofs[i][0][0]
                Rb = Rb @ exp_to_mat(dof[d0:d0 + 3])
            R[i] = Rb
        return P, R


def load_motion(path=MOTION):
    import torch
    d = torch.load(path, map_location="cpu")
    return d.cpu().numpy().astype(np.float64)


def main():
    m = Model()
    data = load_motion()
    print(f"[fk] {len(m.names)} bodies, {m.n_dof} hinge dofs; motion {data.shape}")
    errs = []
    for f in range(0, data.shape[0], 10):
        P, _ = m.fk(data[f, ROOT_POS], exp_to_mat(data[f, ROOT_ROT]), data[f, DOF])
        ref = data[f, BODY].reshape(71, 3)[:len(m.names)]
        errs.append(np.linalg.norm(P - ref, axis=-1))
    errs = np.array(errs)
    hand = [i for i, n in enumerate(m.names) if any(k in n for k in ("CMC", "MCP", "PIP", "DIP", "IP"))]
    print(f"[fk] model FK vs motion joint positions: all bodies median {np.median(errs) * 100:.2f} cm, "
          f"max {errs.max() * 100:.2f} cm; finger bodies median {np.median(errs[:, hand]) * 100:.2f} cm")


if __name__ == "__main__":
    main()
