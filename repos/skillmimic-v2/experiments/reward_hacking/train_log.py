"""Test 0, what the training reward is made of (README.md). CPU only, reads files that already exist.

For each run:
  1. TensorBoard: episode return (= the "training reward" in the baseline docs), episode length, and their ratio
     (per-frame reward). A return that rises only through length means "survives longer", not "imitates better".
  2. Checkpoints (every --every epochs): the ATS bookkeeping saved in env_state.
       motion_time_seqreward[i]  running mean per-frame training reward of episodes that started at frame i+2
       time_sample_rate[i]       probability that a training episode starts at frame i+2
     seq_f2 (start frame 2) is the training's own score for the evaluation start, comparable to the test-1
     column train_mean. ATS moves start frames toward low-reward frames, so the average training reward is taken
     over a different mix of start frames at each epoch (README.md, explanation C).
  3. Joined with the existing deterministic evaluation (experiments/results/eval/det/<run>/), Spearman rank
     correlation of each training signal with lift success, task progress, falls over the evaluated epochs.

    python experiments/reward_hacking/train_log.py                 # every *_fix_* and *_ours_* run with summaries
    python experiments/reward_hacking/train_log.py --runs place_book_fix_n2048_s0
Output: experiments/results/reward_hacking/t0/<run>_{tb,ats,joined}.csv, <run>_ats.npz, correlations.csv
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CK_RE = re.compile(r"_(\d{8})\.pth$")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--runs_dir", default="experiments/runs")
    p.add_argument("--every", type=int, default=50, help="checkpoint stride for the ATS state (epochs)")
    p.add_argument("--out_dir", default="experiments/results/reward_hacking/t0")
    return p.parse_args()


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return float("nan"), len(x)
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation), len(x)


def tb_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    r, l = {}, {}
    for f in sorted(glob.glob(os.path.join(run_dir, "summaries", "*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        for tag, d in (("rewards0/iter", r), ("episode_lengths/iter", l)):
            if tag in tags:
                for s in ea.Scalars(tag):
                    d[int(s.step)] = float(s.value)   # later segments (resume) overwrite earlier ones
    return r, l


def ref_lift_start(clip):
    import torch
    pts = glob.glob(os.path.join(REPO, "skillmimic", "data", "motions", "ParaHome", clip, "*.pt"))
    raw = torch.load(pts[0], map_location="cpu")                 # raw ParaHome layout (paramotion_data_handler)
    z = raw[:, 9 + 180 + 71 * 3 + 2].numpy()                       # object z
    up = z - z[0] > 0.02
    return int(np.argmax(up)) if up.any() else -1, raw.shape[0]


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    a = parse()
    os.chdir(REPO)
    import torch
    runs = a.runs or sorted(os.path.basename(os.path.dirname(p))
                            for p in glob.glob(os.path.join(a.runs_dir, "*_n2048_s*", "summaries")))
    os.makedirs(a.out_dir, exist_ok=True)
    corr_rows = []
    for run in runs:
        rd = os.path.join(a.runs_dir, run)
        if not os.path.isdir(os.path.join(rd, "nn")):
            print(f"[t0] {run}: no such run, skipped")
            continue
        meta = json.load(open(os.path.join(rd, "run_meta.json"))) if os.path.isfile(os.path.join(rd, "run_meta.json")) else {}
        clip = meta.get("clip") or run.split("_")[0] + "_" + run.split("_")[1]
        lift0, L = ref_lift_start(clip)

        # 1. TensorBoard
        r, l = tb_scalars(rd)
        tb = [dict(epoch=e, ep_return=r[e], ep_len=l.get(e, float("nan")),
                   per_frame=r[e] / l[e] if l.get(e) else float("nan")) for e in sorted(r)]
        write_csv(os.path.join(a.out_dir, f"{run}_tb.csv"), tb)

        def tb_at(e, k, win=5):   # mean over epochs e-win..e+win (single iterations are noisy)
            v = [row[k] for row in tb if abs(row["epoch"] - e) <= win]
            return float(np.mean(v)) if v else float("nan")

        # 2. ATS state from checkpoints
        cks = sorted((int(CK_RE.search(p).group(1)), p) for p in glob.glob(os.path.join(rd, "nn", "*.pth"))
                     if CK_RE.search(p))
        evals = {}
        # baseline curve (every 250) first, then the test-1 curve (every 100, same protocol) for the other epochs
        for p in glob.glob(os.path.join("experiments", "results", "eval", "det", run, "*.json")) + \
                glob.glob(os.path.join("experiments", "results", "reward_hacking", "eval", "det", run, "*.json")):
            j = json.load(open(p))
            evals.setdefault(int(j["meta"]["epoch"]), j["aggregate"])
        want = {e for e, _ in cks if e % a.every == 0} | (set(evals) & {e for e, _ in cks})
        ats_rows, seq_m, rate_m, ep_list = [], [], [], []
        for e, p in cks:
            if e not in want:
                continue
            st = torch.load(p, map_location="cpu", weights_only=False).get("env_state") or {}
            if not st.get("motion_time_seqreward"):
                continue
            seq = np.asarray(st["motion_time_seqreward"][0], float)        # index i = start frame i + 2
            rate = np.asarray(st["time_sample_rate"][0], float)
            starts = np.arange(len(seq)) + 2
            ent = float(-(rate * np.log(rate + 1e-12)).sum() / np.log(len(rate)))
            row = dict(epoch=e, seq_mean=float(seq.mean()), seq_f2=float(seq[0]),
                       seq_before_lift=float(seq[starts < lift0].mean()) if lift0 > 2 else float("nan"),
                       seq_after_lift=float(seq[starts >= lift0].mean()) if lift0 > 0 else float("nan"),
                       seq_rate_weighted=float((seq * rate).sum() / rate.sum()),
                       ats_entropy=ent, ats_mass_before_lift=float(rate[starts < lift0].sum()) if lift0 > 2 else float("nan"),
                       ats_mass_first10=float(rate[:10].sum()), ats_peak_start=int(starts[np.argmax(rate)]))
            ats_rows.append(row)
            seq_m.append(seq); rate_m.append(rate); ep_list.append(e)
        write_csv(os.path.join(a.out_dir, f"{run}_ats.csv"), ats_rows)
        if ep_list:
            np.savez_compressed(os.path.join(a.out_dir, f"{run}_ats.npz"), epochs=np.array(ep_list),
                                seqreward=np.stack(seq_m), sample_rate=np.stack(rate_m),
                                start_frames=np.arange(seq_m[0].shape[0]) + 2, lift_start=lift0, clip_frames=L)

        # 3. join with the deterministic evaluation curve
        ats_by = {row["epoch"]: row for row in ats_rows}
        joined = []
        for e in sorted(evals):
            ag = evals[e]
            row = dict(epoch=e, ep_return=tb_at(e, "ep_return"), ep_len=tb_at(e, "ep_len"),
                       per_frame=tb_at(e, "per_frame"))
            row.update({k: ats_by[e][k] for k in ("seq_mean", "seq_f2", "seq_rate_weighted", "ats_entropy",
                                                   "ats_mass_before_lift") if e in ats_by})
            row.update({k: ag.get(k) for k in ("lift_success", "task_success", "task_progress", "fell_in_clip",
                                                "obj_err_mean", "body_err_mean")})
            joined.append(row)
        write_csv(os.path.join(a.out_dir, f"{run}_joined.csv"), joined)
        for sig in ("ep_return", "ep_len", "per_frame", "seq_mean", "seq_f2", "seq_rate_weighted"):
            for tgt in ("lift_success", "task_progress", "fell_in_clip", "obj_err_mean"):
                rho, n = spearman([j_.get(sig, np.nan) for j_ in joined],
                                  [np.nan if j_.get(tgt) is None else j_[tgt] for j_ in joined])
                corr_rows.append(dict(run=run, signal=sig, target=tgt, spearman=rho, n=n))
        print(f"[t0] {run}: {len(tb)} TB epochs, {len(ats_rows)} checkpoints with ATS state, "
              f"{len(joined)} evaluated epochs, reference lift starts at frame {lift0}/{L}")
        if tb:
            f, z = tb[min(range(len(tb)), key=lambda i: abs(tb[i]['epoch'] - 250))], tb[-1]
            print(f"      return {f['ep_return']:.1f} -> {z['ep_return']:.1f}, length {f['ep_len']:.1f} -> {z['ep_len']:.1f}, "
                  f"per-frame {f['per_frame']:.3f} -> {z['per_frame']:.3f} (epoch {f['epoch']} -> {z['epoch']})")
    write_csv(os.path.join(a.out_dir, "correlations.csv"), corr_rows)
    print(f"[t0] wrote {a.out_dir}/ (correlations.csv: training signal vs evaluation, Spearman over evaluated epochs)")


if __name__ == "__main__":
    main()
