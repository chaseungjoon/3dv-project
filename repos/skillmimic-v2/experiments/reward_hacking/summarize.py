"""Tables for tests 1, 4, 5 from the evaluation .json files (README.md). CPU only, a few seconds.

    python experiments/reward_hacking/summarize.py t1     # reward on the evaluation rollouts vs success (+ test 2 factors)
    python experiments/reward_hacking/summarize.py t4     # success and reward by start frame
    python experiments/reward_hacking/summarize.py t5     # 64-env success rates with 95% intervals vs the 8-env curve

Reads <root>/eval/<mode>[_tag]/<run>/*.json, writes <root>/<test>/ (root: experiments/results/reward_hacking).
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = "experiments/results/reward_hacking"
REW_KEYS = ["rew_mean", "rew_sum", "train_return", "train_len", "train_mean",
            "r_body_mean", "r_pos_mean", "r_rot_mean", "r_obj_mean", "r_ig_mean"]
OUT_KEYS = ["lift_success", "task_success", "task_progress", "fell_in_clip", "obj_err_mean", "body_err_mean",
            "paper_success"]


def load(sub):
    """{run: {epoch: json}} for one eval sub-directory (e.g. det, stoch_n64, det_start60)."""
    res = {}
    for p in glob.glob(os.path.join(ROOT, "eval", sub, "*", "*.json")):
        j = json.load(open(p))
        res.setdefault(j["meta"]["run_name"], {})[int(j["meta"]["epoch"])] = j
    return res


def write_csv(path, rows):
    if not rows:
        print(f"[summarize] nothing for {path}")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[summarize] {path} ({len(rows)} rows)")


def rank_corr(x, y, kind="spearman"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return float("nan"), int(len(x))
    from scipy import stats
    r = stats.spearmanr(x, y).correlation if kind == "spearman" else stats.kendalltau(x, y).correlation
    return float(r), int(len(x))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def nz(v):
    return float("nan") if v is None else float(v)


def t1():
    rows, corr, ep_corr = [], [], []
    for mode in ("det", "stoch"):
        for run, by_ep in sorted(load(mode).items()):
            run_rows = []
            for e in sorted(by_ep):
                ag = by_ep[e]["aggregate"]
                if "rew_mean" not in ag:
                    continue
                run_rows.append(dict(run=run, mode=mode, epoch=e, n=ag["n_episodes"],
                                     **{k: ag.get(k) for k in OUT_KEYS + REW_KEYS}))
            rows += run_rows
            for sig in REW_KEYS:
                for tgt in ("lift_success", "task_progress", "fell_in_clip", "obj_err_mean"):
                    rho, n = rank_corr([nz(r[sig]) for r in run_rows], [nz(r[tgt]) for r in run_rows])
                    tau, _ = rank_corr([nz(r[sig]) for r in run_rows], [nz(r[tgt]) for r in run_rows], "kendall")
                    corr.append(dict(run=run, mode=mode, level="checkpoint", signal=sig, target=tgt,
                                     spearman=rho, kendall=tau, discordant_pairs=(1 - tau) / 2 if tau == tau else tau, n=n))
            # episode level: every rollout of every checkpoint pooled
            eps = [ep for e in sorted(by_ep) for ep in by_ep[e]["episodes"] if "rew_mean" in ep]
            for sig in ("rew_mean", "train_mean", "train_return", "r_body_mean", "r_obj_mean", "r_ig_mean"):
                for tgt in ("lift_success", "task_progress", "fell_in_clip"):
                    rho, n = rank_corr([nz(ep.get(sig)) for ep in eps], [nz(ep.get(tgt)) for ep in eps])
                    ep_corr.append(dict(run=run, mode=mode, level="episode", signal=sig, target=tgt, spearman=rho, n=n))
            # mean reward of failed vs successful rollouts
            for tgt in ("lift_success",):
                ok = [nz(ep["rew_mean"]) for ep in eps if ep.get(tgt) == 1]
                bad = [nz(ep["rew_mean"]) for ep in eps if ep.get(tgt) == 0]
                ep_corr.append(dict(run=run, mode=mode, level="episode", signal="rew_mean", target=f"{tgt}=1 vs 0",
                                    mean_success=float(np.nanmean(ok)) if ok else float("nan"),
                                    mean_failure=float(np.nanmean(bad)) if bad else float("nan"),
                                    n_success=len(ok), n_failure=len(bad)))
    write_csv(os.path.join(ROOT, "t1", "curve.csv"), rows)
    write_csv(os.path.join(ROOT, "t1", "correlations.csv"), corr + ep_corr)


def t4():
    rows = []
    for sub in sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "eval", "det_start*"))):
        s = int(re.search(r"start(\d+)", sub).group(1))
        for run, by_ep in sorted(load(sub).items()):
            for e in sorted(by_ep):
                ag = by_ep[e]["aggregate"]
                rows.append(dict(run=run, epoch=e, start_frame=s, clip_frames=by_ep[e]["meta"]["clip_frames"],
                                 **{k: ag.get(k) for k in OUT_KEYS + ["rew_mean", "train_mean", "train_return",
                                                                     "train_len", "r_obj_mean", "r_ig_mean"]}))
    rows.sort(key=lambda r: (r["run"], r["epoch"], r["start_frame"]))
    write_csv(os.path.join(ROOT, "t4", "by_start.csv"), rows)


def t5():
    rows = []
    base = load("det")          # test 1 det 8 env (same checkpoints when both ran)
    old = {}
    for p in glob.glob("experiments/results/eval/det/*/*.json"):   # the baseline 8-env curve (BASELINE_SKILLMIMIC_2.md)
        j = json.load(open(p))
        old.setdefault(j["meta"]["run_name"], {})[int(j["meta"]["epoch"])] = j["aggregate"]
    for sub in ("det_n64", "stoch_n64"):
        for run, by_ep in sorted(load(sub).items()):
            for e in sorted(by_ep):
                j = by_ep[e]
                n = j["aggregate"]["n_episodes"]
                row = dict(run=run, mode=sub, epoch=e, n=n)
                for k in ("lift_success", "task_success", "fell_in_clip", "paper_success"):
                    vals = [ep[k] for ep in j["episodes"]]
                    lo, hi = wilson(sum(vals), n)
                    row.update({k: float(np.mean(vals)), k + "_lo95": lo, k + "_hi95": hi})
                tp = [ep["task_progress"] for ep in j["episodes"]]
                row.update(task_progress=float(np.mean(tp)), task_progress_sd=float(np.std(tp)))
                if "rew_mean" in j["aggregate"]:
                    row.update(rew_mean=j["aggregate"]["rew_mean"], train_mean=j["aggregate"]["train_mean"])
                o = old.get(run, {}).get(e) or (base.get(run, {}).get(e) or {}).get("aggregate")
                if o:
                    row.update(det8_lift_success=o.get("lift_success"), det8_fell_in_clip=o.get("fell_in_clip"))
                rows.append(row)
    write_csv(os.path.join(ROOT, "t5", "n64.csv"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["t1", "t4", "t5"])
    ap.add_argument("--root", default=ROOT, help="results folder of the tests (contains eval/)")
    args = ap.parse_args()
    os.chdir(REPO)
    ROOT = args.root
    {"t1": t1, "t4": t4, "t5": t5}[args.test]()
