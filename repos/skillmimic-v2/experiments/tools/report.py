"""Collect SkillMimic-V2 training logs and evaluation results into one report (no GPU).

    python experiments/tools/report.py                       # experiments/runs -> experiments/results/report
    python experiments/tools/report.py --runs_dir experiments/runs/_check --eval_dir experiments/results/_check/eval \
        --out experiments/results/_check/report

Writes REPORT.md (Korean), runs.csv, final.csv, curve.csv and fig_curve_<clip>.png.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EPOCH_RE = re.compile(r"epoch_num:(\d+) mean_rewards:\[([-\d.eE+na]+)\] fps step: ([\d.]+) fps total: ([\d.]+)")

# arXiv 2505.02094 Table 2 (SR %, RTX 4090, 2048 envs, ~1.0B samples per clip)
PAPER_SR = {
    ("place_book", "sm"): 0.0, ("place_book", "sm_t"): 100.0, ("place_book", "ours"): 100.0,
    ("drink_cup", "sm"): 0.0, ("drink_cup", "sm_t"): 99.9, ("drink_cup", "ours"): 100.0,
    ("place_kettle", "sm"): 0.0, ("place_kettle", "sm_t"): 0.0, ("place_kettle", "ours"): 100.0,
}
PAPER_SAMPLES = 1.0e9
CLIP_KO = {"place_book": "Place Book", "drink_cup": "Drink Cup", "place_kettle": "Place Kettle (T09 비교용)"}
METHOD_KO = {"ours": "SkillMimic-V2 (SM+Ours)", "sm": "SkillMimic (SM)", "sm_t": "SM+T"}


def parse_train_log(path):
    rows = []
    if os.path.isfile(path):
        for line in open(path, errors="ignore"):
            m = EPOCH_RE.search(line)
            if m:
                try:
                    rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
                except ValueError:
                    pass
    return rows


def load_runs(runs_dir):
    runs = {}
    for meta_path in sorted(glob.glob(os.path.join(runs_dir, "*", "run_meta.json"))):
        d = os.path.dirname(meta_path)
        m = json.load(open(meta_path))
        log = parse_train_log(os.path.join(d, "train.log"))
        if log:
            fps = [r[3] for r in log[5:]] or [r[3] for r in log]
            m["epochs_done"] = log[-1][0]
            m["fps_total"] = sum(fps) / len(fps)
            m["s_per_epoch"] = m["num_envs"] * 32 / m["fps_total"]
            m["last_reward"] = log[-1][1]
            m["samples"] = m["epochs_done"] * m["num_envs"] * 32
            m["reward_curve"] = [(r[0], r[1]) for r in log]
        runs[m["run_name"]] = m
    return runs


def load_evals(eval_dir):
    out = defaultdict(list)  # (mode_dir, run) -> [result]
    for p in glob.glob(os.path.join(eval_dir, "*", "*", "*.json")):
        mode_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
        r = json.load(open(p))
        out[(mode_dir, r["meta"]["run_name"])].append(r)
    for k in out:
        out[k].sort(key=lambda r: (r["meta"]["epoch"] or 0))
    return out


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="experiments/runs")
    ap.add_argument("--eval_dir", default="experiments/results/eval")
    ap.add_argument("--out", default="experiments/results/report")
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs_dir)
    evals = load_evals(a.eval_dir)
    keys = ["task_success", "lift_success", "paper_success", "paper_success_clip", "fell_in_clip",
            "task_progress", "obj_err_mean", "body_err_mean", "lift_max", "held_frames"]

    # ---- csv ----
    with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "clip", "method", "num_envs", "seed", "epochs_done", "samples", "fps_total",
                    "s_per_epoch", "wall_h", "peak_gpu_mib", "last_reward"])
        for n, m in runs.items():
            w.writerow([n, m.get("clip"), m.get("method"), m.get("num_envs"), m.get("seed"), m.get("epochs_done"),
                        m.get("samples"), fmt(m.get("fps_total"), 0), fmt(m.get("s_per_epoch"), 2),
                        fmt((m.get("wall_s_total") or 0) / 3600, 2), m.get("peak_gpu_mem_mib"), fmt(m.get("last_reward"), 2)])
    final_rows, curve_rows = [], []
    for (mode_dir, run), rs in sorted(evals.items()):
        for r in rs:
            row = dict(mode=mode_dir, run=run, clip=r["meta"].get("clip"), method=r["meta"].get("method"),
                       epoch=r["meta"]["epoch"], n=r["aggregate"]["n_episodes"])
            row.update({k: r["aggregate"].get(k) for k in keys})
            curve_rows.append(row)
        final_rows.append(curve_rows[-1])
    for name, rows in [("curve.csv", curve_rows), ("final.csv", final_rows)]:
        with open(os.path.join(a.out, name), "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    # ---- figures ----
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        clips = sorted({r["clip"] for r in curve_rows if r["mode"] == "det"} | {m.get("clip") for m in runs.values()} - {None})
        for clip in clips:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            for run, m in runs.items():
                if m.get("clip") != clip:
                    continue
                lab = f"{m.get('method')} n{m.get('num_envs')} s{m.get('seed')}"
                rc = m.get("reward_curve") or []
                if rc:
                    ax[0].plot([e for e, _ in rc], [v for _, v in rc], label=lab, lw=0.8)
                cr = [r for r in curve_rows if r["run"] == run and r["mode"] == "det"]
                if cr:
                    ep = [r["epoch"] for r in cr]
                    ax[1].plot(ep, [r["task_success"] for r in cr], "o-", label=lab + " task")
                    ax[1].plot(ep, [r["lift_success"] for r in cr], "s--", label=lab + " lift")
                    ax[1].plot(ep, [r["paper_success"] for r in cr], "^:", label=lab + " paper")
                    ax[2].plot(ep, [r["obj_err_mean"] for r in cr], "o-", label=lab)
            ax[0].set_title(f"{clip}: mean episode reward (train)"); ax[0].set_xlabel("epoch")
            ax[1].set_title("success (det, 8 envs)"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(-0.05, 1.05)
            ax[2].set_title("object position error [m] (clip frames)"); ax[2].set_xlabel("epoch")
            for x in ax:
                x.grid(alpha=0.3); x.legend(fontsize=7)
            p = os.path.join(a.out, f"fig_curve_{clip}.png")
            fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
            figs.append(os.path.basename(p))
    except Exception as e:  # report must not fail on plotting
        print(f"[report] plot skipped: {e}")

    # ---- REPORT.md ----
    L = ["# SkillMimic-V2 ParaHome baseline 결과 (RTX 5070 12GB)", "",
         f"자동 생성: `experiments/tools/report.py`. 규칙과 지표 정의: `experiments/docs/PROTOCOL.md`.", "",
         "## 1. 학습 run", "",
         "| run | clip | method | env | epoch | 샘플 (논문 대비) | 학습 속도 | s/epoch | 시간 | 최대 VRAM | 마지막 보상 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, m in runs.items():
        s = m.get("samples")
        L.append(f"| `{n}` | {m.get('clip')} | {m.get('method')} | {m.get('num_envs')} | {m.get('epochs_done', '-')} | "
                 f"{(s or 0) / 1e6:.0f}M ({(s or 0) / PAPER_SAMPLES * 100:.0f}%) | {fmt(m.get('fps_total'), 0)} samples/s | "
                 f"{fmt(m.get('s_per_epoch'), 1)} | {fmt((m.get('wall_s_total') or 0) / 3600, 1)} h | "
                 f"{fmt(m.get('peak_gpu_mem_mib'), 0)} MiB | {fmt(m.get('last_reward'), 1)} |")
    L += ["", "## 2. 최종 체크포인트", "",
          "`det` = 결정적 행동 8 env (물리 잡음 반복), `stoch` = 학습 때와 같은 행동 잡음 32 env, "
          "`stoch_perturb` = 물체 시작 위치 교란 (yaw ±45°, xy ≤ 10 cm, 논문 ε-NSR).", "",
          "| run | 평가 | epoch | 과제 성공 (T09 기준) | 들어올림 성공 | 논문 지표 | 논문 SR (참고) | clip 중 넘어짐 | 과제 진행률 | 물체 오차 | 몸 오차 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(final_rows, key=lambda r: (r["clip"] or "", r["method"] or "", r["run"], r["mode"])):
        ps = PAPER_SR.get((r["clip"], r["method"]))
        L.append(f"| `{r['run']}` | {r['mode']} | {r['epoch']} | {fmt(r['task_success'])} | {fmt(r['lift_success'])} | "
                 f"{fmt(r['paper_success'])} | {'-' if ps is None else f'{ps:.0f}%'} | {fmt(r['fell_in_clip'])} | "
                 f"{fmt(r['task_progress'])} | {fmt(r['obj_err_mean'], 3)} m | {fmt(r['body_err_mean'], 3)} m |")
    L += ["", "## 3. 학습 곡선 (det, 8 env)", ""]
    for run in runs:
        cr = [r for r in curve_rows if r["run"] == run and r["mode"] == "det"]
        if not cr:
            continue
        L += [f"**`{run}`**", "", "| epoch | 과제 성공 | 들어올림 성공 | 논문 지표 | 넘어짐 | 과제 진행률 | 물체 오차 | 최대 들어올림 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in cr:
            L.append(f"| {r['epoch']} | {fmt(r['task_success'])} | {fmt(r['lift_success'])} | {fmt(r['paper_success'])} | "
                     f"{fmt(r['fell_in_clip'])} | {fmt(r['task_progress'])} | {fmt(r['obj_err_mean'], 3)} m | {fmt(r['lift_max'], 3)} m |")
        L.append("")
    if figs:
        L += ["## 4. 그림", ""] + [f"![{f}]({f})" for f in figs]
    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"[report] {len(runs)} runs, {len(curve_rows)} evaluations -> {a.out}/REPORT.md")


if __name__ == "__main__":
    main()
