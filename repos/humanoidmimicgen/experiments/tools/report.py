"""HumanoidMimicGen report: training cost and closed-loop success per checkpoint (no GPU).

    python experiments/tools/report.py     # -> experiments/results/report/REPORT.md, success.csv, fig_success.png
"""
import argparse
import csv
import glob
import json
import math
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# arXiv 2605.27724: Table 1 (GR00T N1.6 VLA, 8x H100) and Table 4 (architecture ablation), 100 rollouts each
PAPER = {"02_push_button": {"DP (1,000 HMG demos)": 0.55, "Flow Matching": 1.00, "VLA (GR00T N1.6)": 0.92,
                            "VLA, 100 human demos": 0.82, "VLA, 1 human demo": 0.18}}
STEP_RE = re.compile(r"step:(\d+(?:\.\d+)?)(K?) .*?loss:([\d.]+).*?updt_s:([\d.]+).*?data_s:([\d.]+)")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="experiments/runs")
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--out", default="experiments/results/report")
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)

    L = ["# HumanoidMimicGen (G1, MuJoCo) baseline 결과 (RTX 5070 12GB)", "",
         "자동 생성: `experiments/tools/report.py`. 정책 = 공식 예제 Diffusion Policy (LeRobot, 64/50 chunk, batch 16). "
         "평가 = 공식 `evaluate_policy_example.py` (MuJoCo + WBC closed-loop, seed 0..N-1, 1250 step 안에 성공 여부).", "",
         "## 1. 학습", "", "| run | step | 시간 | step/s | 마지막 loss | 최대 VRAM |", "|---|---|---|---|---|---|"]
    for meta_p in sorted(glob.glob(os.path.join(a.runs_dir, "_logs", "*", "run_meta.json"))):
        d = os.path.dirname(meta_p); m = json.load(open(meta_p))
        steps, losses, upd = [], [], []
        if os.path.isfile(os.path.join(d, "train.log")):
            for line in open(os.path.join(d, "train.log"), errors="ignore"):
                mm = STEP_RE.search(line)
                if mm:
                    steps.append(float(mm.group(1)) * (1000 if mm.group(2) else 1))
                    losses.append(float(mm.group(3))); upd.append(float(mm.group(4)) + float(mm.group(5)))
        sps = (1 / (sum(upd[2:]) / len(upd[2:]))) if len(upd) > 2 else None
        L.append(f"| `{m['run']}` | {int(steps[-1]) if steps else '-'} / {m['steps']} | {(m.get('wall_s_total') or 0) / 3600:.2f} h | "
                 f"{'-' if sps is None else f'{sps:.1f}'} | {losses[-1] if losses else '-'} | {m.get('peak_gpu_mem_mib', '-')} MiB |")

    rows = []
    for p in sorted(glob.glob(os.path.join(a.results_dir, "eval", "*", "eval_*ep.json"))):
        r = json.load(open(p))
        run = os.path.basename(os.path.dirname(p))
        step = int(re.search(r"eval_(\d+)_", os.path.basename(p)).group(1))
        k, n = sum(r["episode_successes"]), r["num_episodes"]
        lo, hi = wilson(k, n)
        logp = p[:-5] + ".log"
        rows.append(dict(run=run, task=r["task_preset"], step=step, n=n, successes=k, success_rate=k / n, ci_lo=lo, ci_hi=hi,
                         mean_steps=sum(r["episode_steps"]) / n))
    with open(os.path.join(a.out, "success.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L += ["", "## 2. Closed-loop 성공률", "", "| run | step | episode | 성공률 | 95% CI (Wilson) | 평균 길이 (step) |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['run']}` | {r['step']} | {r['n']} | **{r['success_rate']:.2f}** | [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | {r['mean_steps']:.0f} |")
    base = []
    for p in sorted(glob.glob(os.path.join(a.results_dir, "baselines", "*ep.json"))):
        r = json.load(open(p)); k, n = sum(r["episode_successes"]), r["num_episodes"]
        lo, hi = wilson(k, n)
        base.append((r["policy"], n, k / n, lo, hi, sum(r["episode_steps"]) / n))
    if base:
        L += ["", "### 학습 없는 기준 정책 (같은 평가 루프, 성공 판정의 바닥값)", "",
              "| 정책 | episode | 성공률 | 95% CI | 평균 길이 |", "|---|---|---|---|---|"]
        desc = {"hold": "hold (가만히 서 있기)", "mean": "mean (평균 행동 반복)", "replay": "replay (시범 행동 열린 루프 재생)",
                "noise": "noise (무작위 행동)"}
        for pol, n, sr, lo, hi, ms in base:
            L.append(f"| {desc.get(pol, pol)} | {n} | {sr:.2f} | [{lo:.2f}, {hi:.2f}] | {ms:.0f} |")
    tasks = sorted({r["task"] for r in rows}) or ["02_push_button"]
    for t in tasks:
        if t in PAPER:
            L += ["", f"논문 값 ({t}, 1,000 demo, 100 rollout): " + ", ".join(f"{k} {v:.2f}" for k, v in PAPER[t].items()) +
                  ". 우리 DP는 공식 예제 설정이라 논문 DP와 학습 설정이 같다는 보장은 없다 (논문은 DP 설정을 밝히지 않음)."]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if rows:
            fig, ax = plt.subplots(figsize=(6, 4))
            for run in sorted({r["run"] for r in rows}):
                rs = sorted([r for r in rows if r["run"] == run], key=lambda r: r["step"])
                x = [r["step"] for r in rs]; y = [r["success_rate"] for r in rs]
                ax.errorbar(x, y, yerr=[[r["success_rate"] - r["ci_lo"] for r in rs], [r["ci_hi"] - r["success_rate"] for r in rs]],
                            fmt="o-", capsize=3, label=run)
            for k, v in PAPER.get(tasks[0], {}).items():
                if k.startswith("DP"):
                    ax.axhline(v, ls="--", c="gray", label=f"paper {k}")
            ax.set_ylim(-0.05, 1.05); ax.set_xlabel("training step"); ax.set_ylabel("success rate"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_success.png"), dpi=110); plt.close(fig)
            L += ["", "![success](fig_success.png)"]
    except Exception as e:
        print(f"[report] plot skipped: {e}")
    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"[report] {len(rows)} evaluations -> {a.out}/REPORT.md")


if __name__ == "__main__":
    main()
