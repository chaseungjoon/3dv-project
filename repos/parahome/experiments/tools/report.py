"""ParaHome x WristMimic report: same T09 method/config on Drink Cup and Place Book vs the T09 kettle baseline.

    python experiments/tools/report.py      # -> experiments/results/report/REPORT.md, curves.csv, fig_curves.png

Reads the T09-format evaluation json of this project (experiments/results/eval/<mode>/<run>/) and, for the
kettle control, the existing T09 baseline evaluations in ../t09-wristmimic/experiments/results/eval/.
The grasp/lift frames of every scene come from experiments/results/audit/clips.csv (audit.sh).
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
T09 = os.path.join(REPO, "..", "t09-wristmimic", "experiments")
T09_BASE_RUN = "default_s0_1004-175106"   # T09 Phase 1 baseline (kettle, 6000 epochs, BASELINE_T09.md)
SCENE_OF_RUN = re.compile(r"^wm_([a-z]+_s\d+)_")
SCENE_CLIP = {"cup_s110": "s110_16_cup_drink", "book_s11": "s11_book_desk2bookshelf", "cup_s89": "s89_2_drink_cup",
              "book_s149": "s149_15_book_bookshelf2table", "kettle_s110": "s110_0_kettle_table2desk"}
LABEL = {"cup_s110": "Drink Cup (s110)", "book_s11": "Place Book (s11)", "cup_s89": "Drink Cup (s89)",
         "book_s149": "Place Book (s149)", "kettle_s110": "Kettle (s110, T09 장면)"}
LABEL_EN = {"cup_s110": "Drink Cup s110", "book_s11": "Place Book s11", "cup_s89": "Drink Cup s89",
            "book_s149": "Place Book s149", "kettle_s110": "Kettle s110 (T09)"}
KEYS = ["official_success", "task_success", "progress", "task_progress", "first_fail_frame", "obj_pos_err_mean",
        "contact_recall", "wrist_pos_err_window", "fell", "first_cause_fall", "first_cause_object", "first_cause_contact",
        "first_cause_wrist_R", "first_cause_wrist_L", "first_cause_body"]


def load(eval_root, run_filter=None, tag=None):
    rows = []
    for p in glob.glob(os.path.join(eval_root, "*", "*", "*.json")):
        mode = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if mode not in ("det", "stoch"):
            continue
        run = os.path.basename(os.path.dirname(p))
        if run_filter and run != run_filter:
            continue
        r = json.load(open(p))
        m = SCENE_OF_RUN.match(run)
        scene = m.group(1) if m else ("kettle_s110" if run == T09_BASE_RUN else None)
        if scene is None:
            continue
        row = dict(source=tag or "parahome", mode=mode, run=run, scene=scene, epoch=r["meta"].get("epoch"),
                   n=r["aggregate"].get("n_episodes"))
        row.update({k: r["aggregate"].get(k) for k in KEYS})
        rows.append(row)
    return rows


def audit_frames():
    out = {}
    p = os.path.join(REPO, "experiments", "results", "audit", "clips.csv")
    if os.path.isfile(p):
        for r in csv.DictReader(open(p)):
            if r["sim_clip"]:
                out[r["sim_clip"]] = r
    return out


def f(x, nd=2):
    return "-" if x is None or x == "" else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="experiments/results/eval")
    ap.add_argument("--out", default="experiments/results/report")
    ap.add_argument("--budget", type=int, default=0, help="compare at this epoch (default: last common epoch)")
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.eval_dir)
    rows += load(os.path.join(T09, "results", "eval"), run_filter=T09_BASE_RUN, tag="t09")
    rows.sort(key=lambda r: (r["scene"], r["run"], r["mode"], r["epoch"] or 0))
    with open(os.path.join(a.out, "curves.csv"), "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    aud = audit_frames()

    runs = defaultdict(list)
    for r in rows:
        runs[(r["run"], r["mode"])].append(r)
    ours_epochs = [max(x["epoch"] for x in rs) for (run, _), rs in runs.items() if run != T09_BASE_RUN]
    budget = a.budget or (min(ours_epochs) if ours_epochs else 0)

    def at(rs, ep):
        c = [x for x in rs if x["epoch"] == ep]
        return c[0] if c else None

    L = ["# ParaHome × WristMimic: T09 방법을 Drink Cup / Place Book에", "",
         "자동 생성: `experiments/tools/report.py`. 방법, 설정, 평가기는 T09 baseline과 같고 **장면만** 다르다 "
         "(`experiments/docs/PROTOCOL.md`).", "",
         f"## 1. 같은 예산 비교 (epoch {budget})", "",
         "`det` = 결정적 행동 8 env, `stoch` = 학습 때 행동 잡음 32 env. 성공 = 시퀀스 끝까지 실패 조건 없음 (T09 공식). "
         "과제 성공 = 물체가 참조에서 10 cm 이내 + 넘어지지 않음. 첫 실패 frame은 잡기/들기 frame(audit)과 같이 본다.", "",
         "| 장면 | 평가 | 공식 성공 | 과제 성공 | 진행률 | 첫 실패 frame | 잡기 / 들기 frame | 물체 오차 | 접촉 재현율 | 손목 오차(창) | 넘어짐 | 첫 원인 (넘어짐/물체/접촉/손목/몸) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, mode), rs in sorted(runs.items(), key=lambda kv: (kv[1][0]["scene"], kv[0][1])):
        r = at(rs, budget) or rs[-1]
        clip = aud.get(SCENE_CLIP.get(r["scene"], ""), {})
        wr = (r.get("first_cause_wrist_R") or 0) + (r.get("first_cause_wrist_L") or 0)
        cause = "/".join(f(r.get(k)) for k in ["first_cause_fall", "first_cause_object", "first_cause_contact"]) + f"/{wr:.2f}/" + f(r.get("first_cause_body"))
        L.append(f"| {LABEL.get(r['scene'], r['scene'])}{' (T09 결과)' if run == T09_BASE_RUN else ''} | {mode} @{r['epoch']} | "
                 f"{f(r['official_success'])} | {f(r['task_success'])} | {f(r['progress'])} | {f(r['first_fail_frame'], 1)} | "
                 f"{clip.get('grasp_frame', '-')} / {clip.get('lift_frame', '-')} | {f(r['obj_pos_err_mean'], 3)} m | "
                 f"{f(r['contact_recall'])} | {f(r['wrist_pos_err_window'], 3)} m | {f(r['fell'])} | {cause} |")
    L += ["", "## 2. 학습 곡선 (det)", ""]
    for (run, mode), rs in sorted(runs.items()):
        if mode != "det":
            continue
        L += [f"**{LABEL.get(rs[0]['scene'], rs[0]['scene'])}** (`{run}`)", "",
              "| epoch | 공식 성공 | 과제 성공 | 진행률 | 첫 실패 frame | 물체 오차 | 접촉 재현율 |", "|---|---|---|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['epoch']} | {f(r['official_success'])} | {f(r['task_success'])} | {f(r['progress'])} | "
                     f"{f(r['first_fail_frame'], 1)} | {f(r['obj_pos_err_mean'], 3)} m | {f(r['contact_recall'])} |")
        L.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for (run, mode), rs in sorted(runs.items()):
            if mode != "det":
                continue
            ep = [r["epoch"] for r in rs]
            lab = LABEL_EN.get(rs[0]["scene"], rs[0]["scene"])
            ax[0].plot(ep, [r["official_success"] for r in rs], "o-", label=lab)
            ax[1].plot(ep, [r["progress"] for r in rs], "o-", label=lab)
            ax[2].plot(ep, [r["first_fail_frame"] for r in rs], "o-", label=lab)
        for x, t in zip(ax, ["official success (det)", "progress (det)", "first failure frame (det)"]):
            x.set_title(t); x.set_xlabel("epoch"); x.grid(alpha=0.3); x.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_curves.png"), dpi=110); plt.close(fig)
        L += ["![curves](fig_curves.png)", ""]
    except Exception as e:
        print(f"[report] plot skipped: {e}")
    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"[report] {len(runs)} run/mode series, budget epoch {budget} -> {a.out}/REPORT.md")


if __name__ == "__main__":
    main()
