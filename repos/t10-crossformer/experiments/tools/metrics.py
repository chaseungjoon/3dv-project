"""Offline metrics for stored CrossFormer predictions (runs/<variant>/<dataset>/pred.npz).

    python experiments/tools/metrics.py                       # every run, own statistics + the stats probes
    python experiments/tools/metrics.py --runs baseline_goal  # one variant

Unnormalization is applied here, so the same predictions can be scored with
  self   the embodiment's own action statistics (correct usage)
  none   no unnormalization (the raw normalized output is sent to the robot)
  <ds>   another dataset's statistics (unit / scale convention of a different robot)

Accuracy metrics use the executed action (chunk step 0) unless the name says otherwise; the episode is the
independent unit (window metrics are averaged per episode first, CIs bootstrap over episodes).
Constraint metrics are computed for the GT actions too ("gt_" prefix) as the reference rate.
Writes experiments/results/metrics/<run>/<dataset>__<stats>.json.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXP, RESULTS, RUNS  # noqa: E402

MOVE_THRESH_M = 0.002  # direction / magnitude metrics only where the GT step moves at least 2 mm
SPEED_FACTOR = 1.5  # speed violation: step translation > 1.5 x the GT p99 step of that embodiment
WORKSPACE_MARGIN_M = 0.05
GRIP_TOL = 0.05


def load_stats():
    return json.load(open(EXP / "checkpoints/crossformer/dataset_statistics.json"))


def load_convention(name):
    p = RESULTS / "conventions" / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run tools/conventions.py {name} first")
    return json.loads(p.read_text())


def unnormalize(pred_norm, stats_sel):
    out = pred_norm.copy()
    if stats_sel is None:
        return out
    mean, std = np.asarray(stats_sel["mean"]), np.asarray(stats_sel["std"])
    mask = np.asarray(stats_sel.get("mask", [True] * len(mean)), bool)
    out[..., mask] = pred_norm[..., mask] * std[mask] + mean[mask]
    return out


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def constraint_flags(act, proprio, A, pos_dims, limits):
    """act: (N,H,7) raw actions, proprio: (N,P). Returns dict of boolean (N,) flags for step 0 and the chunk."""
    lo, hi = limits["min"], limits["max"]
    p01, p99 = limits["p01"], limits["p99"]
    a0 = act[:, 0]
    trans_m = act[..., :3] @ A.T  # (N,H,3)
    step_m = np.linalg.norm(trans_m[:, 0], axis=-1)
    pos = proprio[:, pos_dims]
    cum = pos[:, None] + np.cumsum(trans_m, axis=1)  # EEF positions after each chunk step (lag ignored)
    in_box = lambda x: np.all((x >= limits["ws_lo"]) & (x <= limits["ws_hi"]), axis=-1)  # noqa: E731
    return {
        "v_range": np.any((a0[:, :6] < lo[:6]) | (a0[:, :6] > hi[:6]), axis=-1),
        "v_p01_p99": np.any((a0[:, :6] < p01[:6]) | (a0[:, :6] > p99[:6]), axis=-1),
        "v_speed": step_m > limits["step_max_m"],
        "v_gripper": (a0[:, 6] < -GRIP_TOL) | (a0[:, 6] > 1 + GRIP_TOL),
        "v_workspace_step0": ~in_box(cum[:, 0]),
        "v_workspace_chunk": ~np.all(in_box(cum), axis=1),
    }


def compute(run, name, stats_name, all_stats, dsets):
    d = np.load(RUNS / run / name / "pred.npz")
    meta = json.loads((RUNS / run / name / "meta.json").read_text())
    conv = load_convention(name)
    A = np.asarray(conv["A_action_to_m"])
    pos_dims = dsets[name]["eef_pos_dims"]
    hz = meta["control_hz"]
    own = all_stats[name]["action"]
    sel = None if stats_name == "none" else all_stats[name if stats_name == "self" else stats_name]["action"]

    pred = unnormalize(d["pred_norm"], sel)  # (N,H,7) raw units of the embodiment (if stats are right)
    gt, valid = d["gt"], d["gt_valid"]
    ep = d["episode"]
    std = np.asarray(own["std"])

    # accuracy, executed step
    en = (pred[:, 0, :6] - gt[:, 0, :6]) / std[:6]  # error in the model's normalized space (own stats)
    norm_l1 = np.abs(en).mean(-1)
    raw_mse = ((pred[:, 0, :6] - gt[:, 0, :6]) ** 2).mean(-1)
    pm, gm = pred[:, 0, :3] @ A.T, gt[:, 0, :3] @ A.T
    trans_err_mm = 1000 * np.linalg.norm(pm - gm, axis=-1)
    gnorm, pnorm = np.linalg.norm(gm, axis=-1), np.linalg.norm(pm, axis=-1)
    moving = gnorm > MOVE_THRESH_M
    cos = np.where(moving, (pm * gm).sum(-1) / (np.maximum(pnorm, 1e-9) * np.maximum(gnorm, 1e-9)), np.nan)
    mag_ratio = np.where(moving, pnorm / np.maximum(gnorm, 1e-9), np.nan)
    grip_ok = ((pred[:, 0, 6] > 0.5) == (gt[:, 0, 6] > 0.5)).astype(float)

    # chunk: normalized L1 per horizon step, valid steps only
    en_h = np.abs((pred[..., :6] - gt[..., :6]) / std[:6]).mean(-1)  # (N,H)
    chunk_l1 = [float(en_h[valid[:, h], h].mean()) for h in range(en_h.shape[1])]

    # constraints: limits are always the TRUE embodiment's (that is what the robot can do)
    gt_step = np.linalg.norm(gt[:, 0, :3] @ A.T, axis=-1)
    pos_all = d["proprio"][:, pos_dims]
    limits = {k: np.asarray(own[k]) for k in ("min", "max", "p01", "p99")}
    limits["step_max_m"] = SPEED_FACTOR * float(np.percentile(gt_step, 99))
    limits["ws_lo"] = pos_all.min(0) - WORKSPACE_MARGIN_M
    limits["ws_hi"] = pos_all.max(0) + WORKSPACE_MARGIN_M
    flags = constraint_flags(pred, d["proprio"], A, pos_dims, limits)
    gt_flags = constraint_flags(gt, d["proprio"], A, pos_dims, limits)

    per_window = {
        "norm_l1": norm_l1,
        "raw_mse": raw_mse,
        "trans_err_mm": trans_err_mm,
        "trans_err_mm_s": trans_err_mm * hz,
        "dir_cos": cos,
        "mag_ratio": mag_ratio,
        "grip_acc": grip_ok,
        **{k: v.astype(float) for k, v in flags.items()},
        **{"gt_" + k: v.astype(float) for k, v in gt_flags.items()},
    }
    episodes = np.unique(ep)
    per_ep = {k: np.array([np.nanmean(v[ep == e]) if np.any(~np.isnan(v[ep == e])) else np.nan for e in episodes])
              for k, v in per_window.items()}
    summary = {}
    for k, v in per_ep.items():
        summary[k] = {"mean": float(np.nanmean(v)), "ci95": boot_ci(v)}
    # magnitude ratio is skewed: report the window median as well
    summary["mag_ratio"]["median_windows"] = float(np.nanmedian(mag_ratio))
    summary["norm_l1_per_dim"] = np.abs(en).mean(0).round(4).tolist()
    summary["raw_mae_per_dim"] = np.abs(pred[:, 0] - gt[:, 0]).mean(0).round(6).tolist()
    summary["chunk_norm_l1_by_h"] = [round(x, 4) for x in chunk_l1]
    paired_keys = ("norm_l1", "trans_err_mm_s", "grip_acc", "dir_cos")
    return {
        "run": run,
        "dataset": name,
        "embodiment": dsets[name]["embodiment"],
        "stats": stats_name,
        "control_hz": hz,
        "episodes": int(len(episodes)),
        "windows": int(len(ep)),
        "limits": {
            "step_max_mm": round(1000 * limits["step_max_m"], 2),
            "workspace_lo_m": np.round(limits["ws_lo"], 3).tolist(),
            "workspace_hi_m": np.round(limits["ws_hi"], 3).tolist(),
        },
        "metrics": summary,
        # per-episode values (same episode ids in every variant) for paired comparisons in aggregate.py
        "per_episode": {"episode": episodes.tolist(),
                        **{k: [None if np.isnan(x) else round(float(x), 6) for x in per_ep[k]] for k in paired_keys}},
        "infer_ms_per_window": meta.get("infer_ms_per_window"),
        "peak_gpu_mem_mib": meta.get("peak_gpu_mem_mib"),
    }


def stats_choices(name, run_datasets, probes):
    """Own statistics always; with probes also 'none' and every other evaluated dataset's statistics."""
    out = ["self"]
    if probes:
        out += ["none"] + [o for o in run_datasets if o != name]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run dirs under experiments/runs (default: all)")
    ap.add_argument("--no-probes", action="store_true", help="only own statistics")
    args = ap.parse_args()
    dsets = yaml.safe_load(open(EXP / "configs/datasets.yaml"))
    all_stats = load_stats()
    runs = args.runs or sorted(p.name for p in (RUNS).iterdir() if p.is_dir())
    for run in runs:
        rdir = RUNS / run
        names = sorted(p.name for p in rdir.iterdir() if (p / "pred.npz").exists())
        probes = not args.no_probes and run.startswith("baseline")
        for name in names:
            for s in stats_choices(name, names, probes):
                res = compute(run, name, s, all_stats, dsets)
                out = RESULTS / "metrics" / run / f"{name}__{s}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(res, indent=2))
                m = res["metrics"]
                print(
                    f"{run:22s} {name:22s} stats={s:22s} normL1={m['norm_l1']['mean']:.3f} "
                    f"trans={m['trans_err_mm']['mean']:.1f}mm grip={m['grip_acc']['mean']:.3f} "
                    f"mag={m['mag_ratio']['median_windows']:.2f} v_range={m['v_range']['mean']:.3f} "
                    f"v_ws={m['v_workspace_chunk']['mean']:.3f}"
                )


if __name__ == "__main__":
    main()
