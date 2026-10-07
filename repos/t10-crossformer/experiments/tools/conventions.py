"""Measure each embodiment's action convention from its held-out data (no model involved).

    python experiments/tools/conventions.py bridge_dataset fractal20220817_data taco_play

For every dataset it fits   dpos(t + lag) ~= A @ a_xyz(t) + b   by least squares, where dpos is the change of the
EEF position in the standardized proprio and a_xyz the translation part of the standardized action, for
lag in {-1, 0, +1, +2}. The best lag tells when an action shows up in the state, A gives the action->metre map
(units: singular values; frame: rotation part of the polar decomposition), and R^2 how well a linear
delta-position model explains the data at all. Also reports the gripper convention (+1 open / 0 close).

Writes experiments/results/conventions/{<dataset>.json, CONVENTIONS.md}. metrics.py uses A to express
translation errors in millimetres, which is what makes errors comparable across embodiments.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXP, RESULTS, load_episodes  # noqa: E402

LAGS = (-1, 0, 1, 2)


def fit(a, d):
    X = np.concatenate([a, np.ones((len(a), 1))], 1)
    W, *_ = np.linalg.lstsq(X, d, rcond=None)
    pred = X @ W
    r2 = 1 - ((d - pred) ** 2).sum() / ((d - d.mean(0)) ** 2).sum()
    return W[:3].T, W[3], float(r2)


def polar(A):
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # keep a proper rotation, push the reflection into the scale
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt
    return R, S


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()
    dsets = yaml.safe_load(open(EXP / "configs/datasets.yaml"))
    outdir = RESULTS / "conventions"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.names:
        dc = dsets[name]
        pd = dc["eef_pos_dims"]
        eps = list(load_episodes(name, dc["version"], args.max_episodes, with_images=False))
        res = {"dataset": name, "embodiment": dc["embodiment"], "control_hz": dc["control_hz"], "episodes": len(eps)}
        fits = {}
        for lag in LAGS:
            A_l, D_l = [], []
            for e in eps:
                a, p = e["action"], e["proprio"]
                T = len(a)
                for t in range(T):
                    s0, s1 = t + lag, t + lag + 1
                    if 0 <= s0 and s1 < len(p):
                        A_l.append(a[t, :3])
                        D_l.append(p[s1, pd] - p[s0, pd])
            A_arr, D_arr = np.asarray(A_l), np.asarray(D_l)
            A, b, r2 = fit(A_arr, D_arr)
            fits[lag] = (A, b, r2, len(A_arr))
        best = max(fits, key=lambda k: fits[k][2])
        A, b, r2, n = fits[best]
        R, S = polar(A)
        acts = np.concatenate([e["action"] for e in eps])
        steps_m = np.linalg.norm(np.concatenate([e["action"][:, :3] for e in eps]) @ A.T, axis=1)
        # gripper: does the action lead the measured gripper state, and with which sign?
        g_corr = []
        for e in eps:
            if len(e["action"]) > 3 and e["proprio"].shape[1] > 6:
                ga, gs = e["action"][:-1, 6], e["proprio"][1:, -1]
                if ga.std() > 0 and gs.std() > 0:
                    g_corr.append(np.corrcoef(ga, gs)[0, 1])
        res.update(
            {
                "best_lag": best,
                "r2_by_lag": {str(k): round(v[2], 4) for k, v in fits.items()},
                "A_action_to_m": np.round(A, 6).tolist(),
                "bias_m": np.round(b, 6).tolist(),
                "scale_singular_values": np.round(S, 5).tolist(),
                "frame_rotation_deg": round(rot_angle_deg(R), 2),
                "frame_rotation": np.round(R, 4).tolist(),
                "r2": round(r2, 4),
                "n_transitions": n,
                "action_std_raw": np.round(acts.std(0), 5).tolist(),
                "step_translation_mm": {
                    "median": round(float(np.median(steps_m) * 1000), 2),
                    "p99": round(float(np.percentile(steps_m, 99) * 1000), 2),
                },
                "speed_mm_s_p99": round(float(np.percentile(steps_m, 99) * 1000 * dc["control_hz"]), 1),
                "gripper_action_values": np.unique(np.round(acts[:, 6], 3))[:6].tolist(),
                "gripper_action_vs_next_state_corr": round(float(np.mean(g_corr)), 3) if g_corr else None,
            }
        )
        (outdir / f"{name}.json").write_text(json.dumps(res, indent=2))
        rows.append(res)
        print(json.dumps({k: res[k] for k in ("dataset", "best_lag", "r2_by_lag", "scale_singular_values",
                                               "frame_rotation_deg", "step_translation_mm", "speed_mm_s_p99",
                                               "gripper_action_vs_next_state_corr")}))

    # table over every dataset measured so far, not only this call's
    rows = [json.loads(f.read_text()) for f in sorted(outdir.glob("*.json"))]
    md = [
        "# Action conventions measured from held-out data",
        "",
        "Fit `dpos(t+lag) = A a_xyz(t) + b` (tools/conventions.py). Scale = singular values of A "
        "(metres per action unit); frame = rotation angle of the polar factor of A.",
        "",
        "| dataset | embodiment | Hz | best lag | R^2 | scale (m/unit) | frame rot (deg) | step mm (median / p99) | p99 speed mm/s | gripper corr |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['dataset']} | {r['embodiment']} | {r['control_hz']} | {r['best_lag']} | {r['r2']} | "
            f"{', '.join(f'{x:.4f}' for x in r['scale_singular_values'])} | {r['frame_rotation_deg']} | "
            f"{r['step_translation_mm']['median']} / {r['step_translation_mm']['p99']} | {r['speed_mm_s_p99']} | "
            f"{r['gripper_action_vs_next_state_corr']} |"
        )
    (outdir / "CONVENTIONS.md").write_text("\n".join(md) + "\n")
    print(f"-> {outdir}/CONVENTIONS.md")


if __name__ == "__main__":
    main()
