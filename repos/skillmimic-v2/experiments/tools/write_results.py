"""Write BASELINE_SKILLMIMIC.md (project root) from the three baseline runs. No GPU, a few seconds.

    python experiments/tools/write_results.py            # what write_results.sh runs

Reads experiments/runs/<run>/{run_meta.json, train.log, gpu.csv, nn/} and experiments/results/eval/<mode>/<run>/*.json,
writes ../../BASELINE_SKILLMIMIC.md, experiments/results/figures/baseline_curves.png and
experiments/results/tables/{runs,final,curve}.csv. Works on partial results: unfinished runs are reported with
their status and the command that continues them. Rules and metric definitions: experiments/docs/PROTOCOL.md.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import statistics
import subprocess
from collections import OrderedDict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.abspath(os.path.join(REPO, "..", ".."))
EPOCH_RE = re.compile(r"epoch_num:(\d+) mean_rewards:\[([-\d.eE+na]+)\] fps step: ([\d.]+) fps total: ([\d.]+)")
NUM_RE = re.compile(r"_(\d{8})\.pth$")

CLIPS = OrderedDict([
    ("drink_cup", dict(name="Drink Cup (컵)", script="run_cup.sh", paper_sr=100.0, paper_ensr=33.9)),
    ("place_book", dict(name="Place Book (책)", script="run_book.sh", paper_sr=100.0, paper_ensr=82.4)),
    ("place_kettle", dict(name="Place Kettle (주전자)", script="run_kettle.sh", paper_sr=100.0, paper_ensr=49.9)),
])
MODES = [("det", "결정적 8 env"), ("stoch", "확률적 32 env"), ("stoch_perturb", "확률적 + 물체 교란 32 env")]
PAPER_SAMPLES = 1.0e9
ACTIVE_S = 600  # train.log written within 10 min -> training is (probably) running


def fmt(x, nd=2, pct=False):
    if x is None:
        return "-"
    if pct:
        return f"{x * 100:.0f}%"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def parse_log(path):
    """epoch -> (reward, fps_total). A resumed run repeats the epochs after its last checkpoint; the last
    occurrence is the one that the saved checkpoints continue from."""
    rows = {}
    if os.path.isfile(path):
        for line in open(path, errors="ignore"):
            m = EPOCH_RE.search(line)
            if m:
                try:
                    rows[int(m.group(1))] = (float(m.group(2)), float(m.group(4)))
                except ValueError:
                    pass
    return dict(sorted(rows.items()))


def window_mean(log, center, half=25):
    v = [r for e, (r, _) in log.items() if abs(e - center) <= half]
    return sum(v) / len(v) if v else None


def load_run(d):
    m = json.load(open(os.path.join(d, "run_meta.json")))
    nn = os.path.join(d, "nn")
    numbered = sorted(int(NUM_RE.search(p).group(1)) for p in glob.glob(os.path.join(nn, "*.pth")) if NUM_RE.search(p))
    m["saved_epochs"] = numbered
    m["saved_epoch"] = numbered[-1] if numbered else 0
    m["n_ckpt"] = len(glob.glob(os.path.join(nn, "*.pth")))
    m["ckpt_gb"] = sum(os.path.getsize(p) for p in glob.glob(os.path.join(nn, "*.pth"))) / 1e9
    m["target"] = (m.get("segments") or [{}])[-1].get("target_epochs", m.get("epochs"))
    log_path = os.path.join(d, "train.log")
    m["log"] = parse_log(log_path)
    m["log_age_s"] = (datetime.datetime.now().timestamp() - os.path.getmtime(log_path)) if os.path.isfile(log_path) else None
    fps = [f for e, (_, f) in m["log"].items() if e > 5]
    m["fps_total"] = sum(fps) / len(fps) if fps else None
    m["s_per_epoch"] = m["num_envs"] * 32 / m["fps_total"] if m["fps_total"] else None
    mem, n_gpu = [], 0
    if os.path.isfile(os.path.join(d, "gpu.csv")):
        for line in open(os.path.join(d, "gpu.csv")):
            n_gpu += 1
            try:
                mem.append(float(line.split(",")[1]))
            except Exception:
                pass
    m["peak_gpu_mib"] = max(mem) if mem else None
    segs = m.get("segments", [])
    if segs and all("wall_s" in s for s in segs):
        m["wall_h"], m["wall_est"] = sum(s["wall_s"] for s in segs) / 3600, False
    else:  # a segment ended without bookkeeping (crash, power loss) or is still running: gpu.csv logs every 30 s
        m["wall_h"], m["wall_est"] = n_gpu * 30 / 3600, True
    m["resumes"] = max(len(segs) - 1, 0)
    return m


def load_evals(eval_dir, run):
    out = {}
    for mode, _ in MODES:
        rs = []
        for p in glob.glob(os.path.join(eval_dir, mode, run, "*.json")):
            r = json.load(open(p))
            r["_npz"] = p[:-5] + ".npz"
            rs.append(r)
        out[mode] = sorted(rs, key=lambda r: r["meta"]["epoch"] or 0)
    return out


def episode_frames(r):
    """median first-failure frame of failing episodes, #successes, reference lift-start frame (same step axis)."""
    meta = r["meta"]
    nc = min(meta["clip_frames"] - meta["start_frame"], meta["horizon"])
    fails = [meta["start_frame"] + int(round(e["task_progress"] * nc)) for e in r["episodes"] if not e["task_success"]]
    lift0 = None
    if os.path.isfile(r["_npz"]):
        z = np.load(r["_npz"])["ref_obj"][:nc, 0, 2]
        up = np.nonzero(z - z[0] > 0.02)[0]
        lift0 = meta["start_frame"] + int(up[0]) if len(up) else None
    return (statistics.median(fails) if fails else None), len(r["episodes"]) - len(fails), lift0


def status(clip, m, ev):
    script = CLIPS[clip]["script"]
    if m is None:
        return "시작 안 함", f"`bash {script}`"
    target = m["target"]
    if m["saved_epoch"] < target:
        if m["log_age_s"] is not None and m["log_age_s"] < ACTIVE_S:
            return f"학습 중 ({m['saved_epoch']}/{target} epoch 저장됨)", "끝날 때까지 기다린다"
        return f"중단됨 ({m['saved_epoch']}/{target} epoch 저장됨)", f"`bash {script}` (마지막 체크포인트부터 이어서)"
    final_ok = all(any(r["meta"]["epoch"] == m["saved_epoch"] for r in ev[mode]) for mode, _ in MODES)
    if not final_ok:
        return f"학습 끝 ({m['saved_epoch']} epoch), 평가 남음", f"`bash {script}` (학습은 건너뛰고 평가만)"
    return f"완료 ({m['saved_epoch']} epoch, 평가 완료)", "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="experiments/runs")
    ap.add_argument("--eval_dir", default="experiments/results/eval")
    ap.add_argument("--out", default=os.path.join(ROOT, "BASELINE_SKILLMIMIC.md"))
    ap.add_argument("--fig_dir", default="experiments/results/figures")
    ap.add_argument("--table_dir", default="experiments/results/tables")
    a = ap.parse_args()
    os.chdir(REPO)
    out_dir = os.path.dirname(os.path.abspath(a.out))

    runs, evals = {}, {}
    for p in sorted(glob.glob(os.path.join(a.runs_dir, "*", "run_meta.json"))):
        m = load_run(os.path.dirname(p))
        if m.get("clip") in CLIPS and m.get("method") == "ours":
            runs[m["clip"]] = m  # one run per clip (seed 0); a later name wins
            evals[m["clip"]] = load_evals(a.eval_dir, m["run_name"])

    # ---------------- per-clip summaries ----------------
    S = {}
    for clip in CLIPS:
        m, ev = runs.get(clip), evals.get(clip, {mode: [] for mode, _ in MODES})
        st, todo = status(clip, m, ev)
        final = {}
        if m:
            for mode, _ in MODES:
                rs = [r for r in ev[mode] if r["meta"]["epoch"] == m["saved_epoch"]] or ev[mode][-1:]
                if rs:
                    final[mode] = rs[-1]
        S[clip] = dict(m=m, ev=ev, status=st, todo=todo, final=final)

    # ---------------- csv ----------------
    os.makedirs(a.table_dir, exist_ok=True)
    keys = ["task_success", "lift_success", "paper_success", "fell_in_clip", "task_progress",
            "obj_err_mean", "body_err_mean", "lift_max", "held_frames"]
    with open(os.path.join(a.table_dir, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "run", "status", "saved_epoch", "target", "samples", "fps_total", "s_per_epoch", "wall_h",
                    "peak_gpu_mib", "resumes", "checkpoints", "checkpoint_gb"])
        for clip, s in S.items():
            m = s["m"]
            if m:
                w.writerow([clip, m["run_name"], s["status"], m["saved_epoch"], m["target"], m["saved_epoch"] * m["num_envs"] * 32,
                            fmt(m["fps_total"], 0), fmt(m["s_per_epoch"]), fmt(m["wall_h"]), m["peak_gpu_mib"],
                            m["resumes"], m["n_ckpt"], fmt(m["ckpt_gb"])])
    for name, pick in [("final.csv", "final"), ("curve.csv", "curve")]:
        rows = []
        for clip, s in S.items():
            src = [(mode, r) for mode, r in s["final"].items()] if pick == "final" else [("det", r) for r in s["ev"]["det"]]
            for mode, r in src:
                row = dict(clip=clip, mode=mode, epoch=r["meta"]["epoch"], n=r["aggregate"]["n_episodes"])
                row.update({k: r["aggregate"].get(k) for k in keys})
                rows.append(row)
        with open(os.path.join(a.table_dir, name), "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    # ---------------- figure ----------------
    fig_rel = None
    if any(s["m"] for s in S.values()):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(len(CLIPS), 3, figsize=(15, 3.6 * len(CLIPS)), squeeze=False)
            for i, (clip, s) in enumerate(S.items()):
                m = s["m"]
                if m and m["log"]:
                    ax[i, 0].plot(list(m["log"]), [r for r, _ in m["log"].values()], lw=0.6)
                cr = s["ev"]["det"]
                if cr:
                    ep = [r["meta"]["epoch"] for r in cr]
                    for k, sty in [("task_success", "o-"), ("lift_success", "s--"), ("paper_success", "^:")]:
                        ax[i, 1].plot(ep, [r["aggregate"][k] for r in cr], sty, label=k)
                    ax[i, 2].plot(ep, [r["aggregate"]["obj_err_mean"] for r in cr], "o-")
                ax[i, 0].set_title(f"{clip}: train mean episode reward")
                ax[i, 1].set_title(f"{clip}: success (det, 8 envs)"); ax[i, 1].set_ylim(-0.05, 1.05)
                ax[i, 2].set_title(f"{clip}: object position error [m]")
                for x in ax[i]:
                    x.set_xlabel("epoch"); x.grid(alpha=0.3)
                if cr:
                    ax[i, 1].legend(fontsize=7)
            os.makedirs(a.fig_dir, exist_ok=True)
            fig_path = os.path.join(a.fig_dir, "baseline_curves.png")
            fig.tight_layout(); fig.savefig(fig_path, dpi=100); plt.close(fig)
            fig_rel = os.path.relpath(os.path.abspath(fig_path), out_dir)
        except Exception as e:  # the document must not fail on plotting
            print(f"[results] figure skipped: {e}")

    # ---------------- decision (PROTOCOL.md 5, fixed before results) ----------------
    complete = all(s["m"] and "det" in s["final"] and s["m"]["saved_epoch"] >= s["m"]["target"] for s in S.values())
    best, trend = 0.0, {}
    for clip, s in S.items():
        d = s["final"].get("det")
        if d:
            best = max(best, d["aggregate"]["task_success"], d["aggregate"]["lift_success"])
        m = s["m"]
        if m and m["log"] and m["saved_epoch"] >= 1250:
            e1, e0 = m["saved_epoch"], m["saved_epoch"] - 1000
            r1, r0 = window_mean(m["log"], e1 - 125, 125), window_mean(m["log"], e0, 125)
            cr = {r["meta"]["epoch"]: r["aggregate"]["task_progress"] for r in s["ev"]["det"]}
            dp = (cr[e1] - cr[e0]) if (e1 in cr and e0 in cr) else None
            rel = (r1 - r0) / abs(r0) if (r1 is not None and r0) else None
            trend[clip] = dict(r0=r0, r1=r1, rel=rel, dp=dp, e0=e0, e1=e1,
                               rising=bool((rel is not None and rel >= 0.01) or (dp is not None and dp >= 0.05)))

    # ---------------- markdown ----------------
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    L = ["# SkillMimic-V2 Baseline 결과: Drink Cup / Place Book / Place Kettle", "",
         f"자동 생성 {now} · `repos/skillmimic-v2/write_results.sh` · 코드 `{git_commit()}` · "
         "계획 `PROPOSAL_T09.md` · 규칙과 지표 `repos/skillmimic-v2/experiments/docs/PROTOCOL.md`", "",
         "> 이 파일은 손으로 고치지 않는다. 다시 만들려면 `cd repos/skillmimic-v2 && bash write_results.sh`.", "",
         "## 0. 진행 상태", "",
         "| clip | 상태 | 다음 할 일 |", "|---|---|---|"]
    for clip, s in S.items():
        L.append(f"| {CLIPS[clip]['name']} | {s['status']} | {s['todo']} |")

    L += ["", "## 1. 요약 (최종 체크포인트, 결정적 8 env)", "",
          "| clip | epoch | 과제 성공 | 들어올림 성공 | 논문 지표 | 논문 SR (4090, 1.0B 샘플) | clip 중 넘어짐 | 과제 진행률 | 물체 오차 | 실패 frame 중앙값 | 참조 들기 시작 frame |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for clip, s in S.items():
        d = s["final"].get("det")
        if not d:
            L.append(f"| {CLIPS[clip]['name']} | - | - | - | - | {CLIPS[clip]['paper_sr']:.0f}% | - | - | - | - | - |")
            continue
        g = d["aggregate"]
        ff, nsucc, lift0 = episode_frames(d)
        L.append(f"| {CLIPS[clip]['name']} | {d['meta']['epoch']} | {fmt(g['task_success'], pct=True)} | {fmt(g['lift_success'], pct=True)} | "
                 f"{fmt(g['paper_success'], pct=True)} | {CLIPS[clip]['paper_sr']:.0f}% | {fmt(g['fell_in_clip'], pct=True)} | "
                 f"{fmt(g['task_progress'])} | {fmt(g['obj_err_mean'], 3)} m | "
                 f"{'- (전부 성공)' if ff is None else f'{ff:.0f}'} | {fmt(lift0)} |")
    L += ["", "- 과제 성공: clip의 모든 frame에서 물체가 참조 위치 10 cm 안 **그리고** 넘어지지 않음 (T09 `task_success`와 같은 정의). "
          "들어올림 성공: 참조 들어올림 높이의 50% 이상 들고 그때 손목 0.2 m 안. 논문 지표: 업스트림 metric (참고용).",
          "- 실패 frame: 처음으로 물체 오차 10 cm 초과 또는 넘어진 frame (실패한 episode만). 참조 들기 시작: 참조 물체가 2 cm 넘게 올라간 첫 frame. "
          "두 값은 같은 시간축이라 실패가 **들기 전/중/후** 어디서 나는지 바로 비교된다.", ""]

    L += ["## 2. 판정 (PROTOCOL.md 5절, 결과를 보기 전에 정한 규칙을 그대로 적용)", ""]
    if not complete:
        L.append("**잠정**: 아직 끝나지 않은 run이 있다. 세 run이 모두 끝나고 평가까지 되면 확정된다.")
        L.append("")
    if best >= 0.5:
        L.append(f"- 결과: clip 하나 이상에서 과제 성공 또는 들어올림 성공 ≥ 0.5 (최고 {best:.2f}).")
        L.append("- 해석: **12GB/RTX 5070은 이 종류의 과제(ParaHome 물체 잡고 들기)에 근본적 장애가 아니다.** "
                 "T09(WristMimic) 0%의 주원인은 하드웨어가 아니라 방법/학습 구조 쪽이다 (H-task).")
    elif trend:
        rising = [c for c, t in trend.items() if t["rising"]]
        L.append(f"- 결과: 세 clip 모두 과제 성공·들어올림 성공 < 0.5 (최고 {best:.2f}).")
        if rising:
            L.append(f"- 학습 곡선이 마지막 1000 epoch 동안 계속 오름: {', '.join(rising)} → **예산 부족**. "
                     "하드웨어 '속도' 문제일 수는 있으나 '메모리' 문제는 아니다 (2048 env가 들어갔으므로). 예산을 늘려 이어서 학습할 수 있다 (`EPOCHS=6000 bash run_<clip>.sh`).")
        else:
            L.append("- 학습 곡선이 평평함 → 이 GPU 설정에서 이 종류의 과제가 막힌다는 증거 (H-hw 쪽). 버퍼/물리 설정 점검이 필요하다.")
    else:
        L.append("- 아직 판정할 결과가 없다.")
    if trend:
        L += ["", "곡선 추세 (오름 = 보상 +1% 이상 **또는** 과제 진행률 +0.05 이상, 마지막 1000 epoch 기준):", "",
              "| clip | 구간 | 학습 보상 (전 → 후) | 보상 변화 | 과제 진행률 변화 (det) | 추세 |", "|---|---|---|---|---|---|"]
        for clip, t in trend.items():
            rel = "-" if t["rel"] is None else f"{t['rel'] * 100:+.1f}%"
            L.append(f"| {CLIPS[clip]['name']} | {t['e0']} → {t['e1']} | {fmt(t['r0'])} → {fmt(t['r1'])} | "
                     f"{rel} | {fmt(t['dp'])} | {'오름' if t['rising'] else '평평'} |")
    L.append("")

    L += ["## 3. 학습 run", "",
          "| clip | 저장된 epoch / 목표 | 샘플 (논문 1.0B 대비) | 학습 속도 | s/epoch | 학습 시간 | 최대 VRAM | 이어 학습 | 체크포인트 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for clip, s in S.items():
        m = s["m"]
        if not m:
            L.append(f"| {CLIPS[clip]['name']} | - | - | - | - | - | - | - | - |")
            continue
        smp = m["saved_epoch"] * m["num_envs"] * 32
        L.append(f"| {CLIPS[clip]['name']} | {m['saved_epoch']} / {m['target']} | {smp / 1e6:.0f}M ({smp / PAPER_SAMPLES * 100:.0f}%) | "
                 f"{fmt(m['fps_total'], 0)} samples/s | {fmt(m['s_per_epoch'], 1)} | {fmt(m['wall_h'], 1)} h{' (추정)' if m['wall_est'] else ''} | "
                 f"{fmt(m['peak_gpu_mib'], 0)} MiB | {m['resumes']}회 | {m['n_ckpt']}개, {m['ckpt_gb']:.1f} GB |")
    L += ["", f"설정: SkillMimic-V2 (SM + STF + ATS + HE + buffer node, STG 없음), 2048 env, seed 0, PPO는 논문 표 8과 같음. "
          "학습 시간에 '(추정)'이 붙으면 기록 없이 끝난 구간(강제 종료, 진행 중)이 있어 GPU 로그 길이로 셌다.", ""]

    L += ["## 4. 최종 체크포인트 상세", "",
          f"논문 ε-NSR (SM + Ours): Drink Cup {CLIPS['drink_cup']['paper_ensr']}%, Place Book {CLIPS['place_book']['paper_ensr']}%, "
          f"Place Kettle {CLIPS['place_kettle']['paper_ensr']}%. 우리 교란 평가는 확률적 행동이라 논문(결정적)보다 엄격하다.", "",
          "| clip | 평가 | epoch | n | 과제 성공 | 들어올림 성공 | 논문 지표 | clip 중 넘어짐 | 과제 진행률 | 물체 오차 | 몸 오차 | 최대 들어올림 | 들고 있던 frame |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for clip, s in S.items():
        for mode, label in MODES:
            r = s["final"].get(mode)
            if not r:
                continue
            g = r["aggregate"]
            L.append(f"| {CLIPS[clip]['name']} | {label} | {r['meta']['epoch']} | {g['n_episodes']} | {fmt(g['task_success'])} | "
                     f"{fmt(g['lift_success'])} | {fmt(g['paper_success'])} | {fmt(g['fell_in_clip'])} | {fmt(g['task_progress'])} | "
                     f"{fmt(g['obj_err_mean'], 3)} m | {fmt(g.get('body_err_mean'), 3)} m | {fmt(g['lift_max'], 3)} m | {fmt(g['held_frames'], 0)} |")
    L.append("")

    L += ["## 5. 학습 곡선 (결정적 8 env)", ""]
    any_curve = False
    for clip, s in S.items():
        cr = s["ev"]["det"]
        if not cr:
            continue
        any_curve = True
        L += [f"**{CLIPS[clip]['name']}**", "",
              "| epoch | 학습 보상 | 과제 성공 | 들어올림 성공 | 논문 지표 | 넘어짐 | 과제 진행률 | 물체 오차 | 최대 들어올림 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in cr:
            g, e = r["aggregate"], r["meta"]["epoch"]
            L.append(f"| {e} | {fmt(window_mean(s['m']['log'], e) if s['m'] else None)} | {fmt(g['task_success'])} | {fmt(g['lift_success'])} | "
                     f"{fmt(g['paper_success'])} | {fmt(g['fell_in_clip'])} | {fmt(g['task_progress'])} | {fmt(g['obj_err_mean'], 3)} m | "
                     f"{fmt(g['lift_max'], 3)} m |")
        L.append("")
    if not any_curve:
        L += ["(아직 평가된 체크포인트가 없다)", ""]
    if fig_rel:
        L += ["![학습 곡선](" + fig_rel + ")", ""]

    L += ["## 6. 읽을 때 주의", "",
          "- **논문 재현이 아니다.** 예산은 clip당 3000 epoch = 1.97억 샘플로 논문(약 1.0B)의 20%다 (T09 WristMimic baseline과 같은 샘플 수). GPU는 RTX 5070 12GB.",
          "- 결정적 평가도 GPU PhysX가 비트 단위로 재현되지 않아 8 env를 돌린다. 8 env의 해상도는 0.125이므로 그보다 작은 차이는 '차이 없음'으로 본다.",
          "- 학습에서 episode는 넘어질 때만 끝나고 60 frame마다 랜덤 frame에서 다시 시작한다. 평가는 frame 2에서 시작해 300 step, 처음 넘어지면 그 env는 끝.",
          "- 시뮬레이션에서 책은 참조보다 약 3.8 cm 높게 놓인다 (충돌 형상). Place Kettle 참조 자체가 논문 지표의 '60 frame 이상' 조건을 45 frame만 만족한다 (CODE_NOTES.md 4절).",
          "- 학습 clip = 평가 clip. 일반화나 실제 로봇 성공을 주장하지 않는다.", "",
          "## 7. 원자료와 재현", "",
          "| 무엇 | 위치 (`repos/skillmimic-v2/` 기준) |", "|---|---|",
          "| 체크포인트 (10 epoch마다), train.log, gpu.csv, run_meta.json | `experiments/runs/<clip>_ours_n2048_s0/` (git 제외) |",
          "| 체크포인트별 env별 지표, frame별 궤적 | `experiments/results/eval/<det|stoch|stoch_perturb>/<run>/*.json, *.npz` |",
          "| 표 (csv), 그림 | `experiments/results/tables/`, `experiments/results/figures/` |", "",
          "```bash", "cd repos/skillmimic-v2", "bash run_cup.sh && bash run_book.sh && bash run_kettle.sh   # 각각 끊겨도 같은 명령으로 이어서",
          "bash write_results.sh                                       # 이 파일", "```", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(L))
    print(f"[results] {sum(1 for s in S.values() if s['m'])} run(s) -> {a.out}")
    for clip, s in S.items():
        print(f"[results]   {clip}: {s['status']}")


if __name__ == "__main__":
    main()
