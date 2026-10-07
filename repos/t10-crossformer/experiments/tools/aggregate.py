"""Collect metrics / closed-loop results into tables, figures and REPORT.md (no GPU, seconds).

    python experiments/tools/aggregate.py

Reads experiments/results/{metrics,sim,conventions} and runs/*/<dataset>/pred.npz (for the trajectory figures).
Writes experiments/results/report/{REPORT.md, offline.csv, sim.csv, fig_*.png}.
"""

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import RESULTS, RUNS  # noqa: E402
from metrics import load_stats, unnormalize  # noqa: E402

RES = RESULTS
OUT = RES / "report"
EMB_NAME = {"bridge_dataset": "WidowX (Bridge)", "fractal20220817_data": "Google Robot (RT-1)",
            "taco_play": "Franka (TACO)"}
PHASE = {
    "Phase 1 baseline": ["baseline_lang", "baseline_goal"],
    "Phase 2 conditioning / history": ["baseline_goal", "cond_goal_nb", "cond_goal_sub8", "cond_none", "baseline_lang",
                                       "hist_w1", "hist_w2"],
    "Phase 3 observation / rate probes": ["baseline_goal", "obs_center_crop", "obs_hflip", "rate_x2", "rate_x3"],
}
SHOW = [("norm_l1", "norm L1", "{:.3f}"), ("trans_err_mm", "trans err mm/step", "{:.1f}"),
        ("trans_err_mm_s", "trans err mm/s", "{:.0f}"), ("dir_cos", "dir cos", "{:.2f}"),
        ("grip_acc", "gripper acc", "{:.3f}")]
VIOL = [("v_p01_p99", "outside train p01-p99"), ("v_range", "out of train range"), ("v_speed", "speed > 1.5x p99"),
        ("v_workspace_chunk", "leaves workspace (chunk)"), ("v_gripper", "gripper outside [0,1]")]


def load_metrics():
    rows = {}
    for f in sorted((RES / "metrics").glob("*/*.json")):
        r = json.loads(f.read_text())
        rows[(r["run"], r["dataset"], r["stats"])] = r
    return rows


def fmt_ci(m, key, f="{:.3f}"):
    v = m[key]
    lo, hi = v["ci95"]
    return f"{f.format(v['mean'])} [{f.format(lo)}, {f.format(hi)}]"


def offline_csv(rows):
    keys = ["norm_l1", "raw_mse", "trans_err_mm", "trans_err_mm_s", "dir_cos", "mag_ratio", "grip_acc"] + \
           [k for k, _ in VIOL] + ["gt_" + k for k, _ in VIOL]
    with open(OUT / "offline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "dataset", "embodiment", "stats", "episodes", "windows", "control_hz"] + keys +
                   ["mag_ratio_median", "ms_per_window"])
        for (run, ds, st), r in sorted(rows.items()):
            m = r["metrics"]
            w.writerow([run, ds, r["embodiment"], st, r["episodes"], r["windows"], r["control_hz"]] +
                       [round(m[k]["mean"], 6) for k in keys] +
                       [round(m["mag_ratio"]["median_windows"], 4), r.get("infer_ms_per_window")])


def table_variants(rows, runs, datasets):
    lines = ["| variant | embodiment | episodes | " + " | ".join(n for _, n, _ in SHOW) + " | mag ratio (median) |",
             "|---|---|---|" + "---|" * len(SHOW) + "---|"]
    for run in runs:
        for ds in datasets:
            r = rows.get((run, ds, "self"))
            if r is None:
                continue
            m = r["metrics"]
            cells = [fmt_ci(m, k, f) if k == "norm_l1" else f.format(m[k]["mean"]) for k, _, f in SHOW]
            lines.append(f"| {run} | {EMB_NAME.get(ds, ds)} | {r['episodes']} | " + " | ".join(cells) +
                         f" | {m['mag_ratio']['median_windows']:.2f} |")
    return lines


def table_violations(rows, runs, datasets, stats="self"):
    lines = ["| variant | embodiment | stats | " + " | ".join(n for _, n in VIOL) + " |",
             "|---|---|---|" + "---|" * len(VIOL)]
    for run in runs:
        for ds in datasets:
            r = rows.get((run, ds, stats))
            if r is None:
                continue
            m = r["metrics"]
            cells = [f"{100 * m[k]['mean']:.1f}% (GT {100 * m['gt_' + k]['mean']:.1f}%)" for k, _ in VIOL]
            lines.append(f"| {run} | {EMB_NAME.get(ds, ds)} | {stats} | " + " | ".join(cells) + " |")
    return lines


def table_stats_probe(rows, run, datasets):
    lines = ["| embodiment | unnormalized with | norm L1 | trans err mm/step | mag ratio | out of train range | "
             "leaves workspace |", "|---|---|---|---|---|---|---|"]
    order = lambda st: (st != "self", st != "none", st)  # noqa: E731
    for ds in datasets:
        for st in sorted({k[2] for k in rows if k[0] == run and k[1] == ds}, key=order):
            r = rows[(run, ds, st)]
            m = r["metrics"]
            label = {"self": "own statistics (correct)", "none": "nothing (raw normalized output)"}.get(
                st, f"{EMB_NAME.get(st, st)} statistics")
            lines.append(f"| {EMB_NAME.get(ds, ds)} | {label} | {m['norm_l1']['mean']:.3f} | "
                         f"{m['trans_err_mm']['mean']:.1f} | {m['mag_ratio']['median_windows']:.2f} | "
                         f"{100 * m['v_range']['mean']:.1f}% | {100 * m['v_workspace_chunk']['mean']:.1f}% |")
    return lines


def table_unit_dominance(rows, run, datasets):
    """Why raw MSE cannot be pooled across robots: share of the pooled raw MSE per embodiment."""
    got = [(ds, rows[(run, ds, "self")]) for ds in datasets if (run, ds, "self") in rows]
    if len(got) < 2:
        return []
    raw = {ds: r["metrics"]["raw_mse"]["mean"] for ds, r in got}
    nrm = {ds: r["metrics"]["norm_l1"]["mean"] for ds, r in got}
    mm = {ds: r["metrics"]["trans_err_mm"]["mean"] for ds, r in got}
    lines = ["| embodiment | raw MSE (dataset units) | share of pooled raw MSE | norm L1 | share | "
             "trans err mm | share |", "|---|---|---|---|---|---|---|"]
    for ds, _ in got:
        lines.append(f"| {EMB_NAME.get(ds, ds)} | {raw[ds]:.2e} | {100 * raw[ds] / sum(raw.values()):.1f}% | "
                     f"{nrm[ds]:.3f} | {100 * nrm[ds] / sum(nrm.values()):.1f}% | {mm[ds]:.1f} | "
                     f"{100 * mm[ds] / sum(mm.values()):.1f}% |")
    return lines


def table_paired(rows, runs, datasets, ref="baseline_goal"):
    """Variant minus reference on the same episodes (frame_skip variants have the same episodes too)."""
    from metrics import boot_ci

    lines = [f"| variant | embodiment | paired episodes | d norm L1 vs {ref} [95% CI] | d trans err mm/s [95% CI] |",
             "|---|---|---|---|---|"]
    for run in runs:
        if run == ref:
            continue
        for ds in datasets:
            a, b = rows.get((run, ds, "self")), rows.get((ref, ds, "self"))
            if a is None or b is None:
                continue
            ea, eb = a["per_episode"], b["per_episode"]
            common = sorted(set(ea["episode"]) & set(eb["episode"]))
            ia = {e: i for i, e in enumerate(ea["episode"])}
            ib = {e: i for i, e in enumerate(eb["episode"])}
            cells = []
            for k, f in (("norm_l1", "{:+.3f}"), ("trans_err_mm_s", "{:+.1f}")):
                d = np.array([np.nan if ea[k][ia[e]] is None or eb[k][ib[e]] is None else ea[k][ia[e]] - eb[k][ib[e]]
                              for e in common])
                lo, hi = boot_ci(d)
                mark = " *" if (lo > 0 or hi < 0) else ""
                cells.append(f"{f.format(np.nanmean(d))} [{f.format(lo)}, {f.format(hi)}]{mark}")
            lines.append(f"| {run} | {EMB_NAME.get(ds, ds)} | {len(common)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("`*` = CI가 0을 포함하지 않음 (PROTOCOL.md 3절의 '차이 있음' 기준).")
    return lines


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def load_sim():
    out = {}
    for f in sorted((RES / "sim").glob("*/*.json")):
        r = json.loads(f.read_text())
        out[(r["variant"], r["suite"])] = r
    return out


def sim_tables(sim):
    if not sim:
        return ["(아직 closed-loop 결과 없음: `bash experiments/scripts/phase4_closed_loop.sh`)"]
    lines = ["| variant | suite | embodiment | task | success [Wilson 95% CI] | n |", "|---|---|---|---|---|---|"]
    with open(OUT / "sim.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "suite", "embodiment", "task", "success_rate", "n", "partial"])
        for (v, s), r in sorted(sim.items()):
            for task, t in r["per_task"].items():
                part = " (partial)" if r.get("partial") else ""
                lo, hi = wilson(round(t["success_rate"] * t["n"]), t["n"])
                lines.append(f"| {v}{part} | {s} | {r['embodiment']} | {task} | {100 * t['success_rate']:.1f}% "
                             f"[{100 * lo:.0f}, {100 * hi:.0f}] | {t['n']} |")
                w.writerow([v, s, r["embodiment"], task, t["success_rate"], t["n"], r.get("partial")])
            lines.append(f"| {v} | {s} | {r['embodiment']} | **mean over tasks** | "
                         f"**{100 * r['mean_success']:.1f}%** | {r['episodes']} |")
    return lines


# SimplerEnv visual-matching numbers reported for Octo-Base in the SimplerEnv paper (Li et al., 2024) and README.
# Reference only: re-check against the paper table before quoting. Bridge numbers are averages over init_rng 0/2/4.
OCTO_BASE_PUBLISHED = {
    "PutSpoonOnTableClothInScene-v0": 0.125, "PutCarrotOnPlateInScene-v0": 0.083,
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0": 0.000, "PutEggplantInBasketScene-v0": 0.431,
    "coke_can": 0.170, "move_near": 0.042,
}
BRIDGE_TASKS = ["PutSpoonOnTableClothInScene-v0", "PutCarrotOnPlateInScene-v0",
                "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutEggplantInBasketScene-v0"]
SHORT = {"PutSpoonOnTableClothInScene-v0": "spoon", "PutCarrotOnPlateInScene-v0": "carrot",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "stack", "PutEggplantInBasketScene-v0": "eggplant"}
GRASP_KEY = {"bridge": "is_src_obj_grasped", "coke_can": "grasped", "move_near": "moved_correct_obj"}


def sim_episodes(sim, variant, family, only_first_urdf):
    """{episode key: (success, stage flag)} for one variant and suite family (bridge / coke_can / move_near),
    taken from the full or the *_quick result. Keys identify the same initial state across variants."""
    for suite in ([family] if family == "bridge" else [family + "_quick", family]):
        r = sim.get((variant, suite))
        if r is None:
            continue
        out = {}
        for j in r["jobs"]:
            if only_first_urdf and j["urdf_version"] != "None":
                continue
            stats = j.get("episode_stats") or [{} for _ in j["success"]]
            for i, (succ, st) in enumerate(zip(j["success"], stats)):
                key = (j["task"], j["env_name"], j["urdf_version"], tuple(j["robot_init"]), i)
                out[key] = (bool(succ), bool(st.get(GRASP_KEY[family], False)))
        return out
    return None


def mcnemar_p(a, b):
    """Exact two-sided McNemar test on paired booleans."""
    from math import comb

    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def table_screening(sim, ref="sim_baseline"):
    """Closed-loop screening: every variant vs the baseline on the SAME episodes (original URDF for Google)."""
    variants = sorted({v for v, _ in sim}, key=lambda v: (v != ref, v))
    fams = [("bridge", False), ("coke_can", True), ("move_near", True)]
    lines = ["| variant | WidowX success (grasp) | Google coke can success (grasp) | Google move near success "
             "(correct object moved) | paired vs baseline: success p / grasp p |",
             "|---|---|---|---|---|"]
    base = {f: sim_episodes(sim, ref, f, u) for f, u in fams}
    for v in variants:
        cells, ps = [], []
        for f, u in fams:
            cur = sim_episodes(sim, v, f, u)
            if not cur:
                cells.append("-")
                continue
            succ = np.mean([x[0] for x in cur.values()])
            grasp = np.mean([x[1] for x in cur.values()])
            cells.append(f"{100 * succ:.1f}% ({100 * grasp:.0f}%) n={len(cur)}")
            b = base.get(f)
            if v != ref and b:
                common = sorted(set(cur) & set(b))
                if common:
                    p_s = mcnemar_p([b[k][0] for k in common], [cur[k][0] for k in common])
                    p_g = mcnemar_p([b[k][1] for k in common], [cur[k][1] for k in common])
                    ps.append(f"{f}: {p_s:.3f} / {p_g:.3f}")
        lines.append(f"| {v} | " + " | ".join(cells) + f" | {'; '.join(ps) if ps else '-'} |")
    lines += ["", "Google 열은 원래 URDF만 (기준선은 전체 run에서 같은 episode를 뽑음). 괄호 = 잡기(또는 맞는 물체 이동) 단계 도달 비율. "
              "p = 같은 episode끼리의 exact McNemar (p < 0.05면 차이 있음)."]
    return lines


def table_octo_validity(sim):
    runs = sorted({v for v, _ in sim if v.startswith("octo-base_rng")})
    if not runs:
        return ["(아직 없음: `bash experiments/scripts/phase5_sim_validity.sh`)"]
    lines = ["| task | 우리 Octo-Base (seed 수) | 논문 보고값 | 우리 CrossFormer baseline |", "|---|---|---|---|"]

    def rate(variant, family, task=None, first_urdf=True):
        e = sim_episodes(sim, variant, family, first_urdf if family != "bridge" else False)
        if not e:
            return None
        vals = [x[0] for k, x in e.items() if task is None or k[1] == task]
        return float(np.mean(vals)) if vals else None

    for task in BRIDGE_TASKS + ["coke_can", "move_near"]:
        fam, t = ("bridge", task) if task in BRIDGE_TASKS else (task, None)
        ours = [x for x in (rate(r, fam, t) for r in runs) if x is not None]
        cf = rate("sim_baseline", fam, t)
        name = SHORT.get(task, task) + ("" if fam == "bridge" else " (원래 URDF)")
        lines.append(f"| {name} | {100 * np.mean(ours):.1f}% ({len(ours)}) | {100 * OCTO_BASE_PUBLISHED[task]:.1f}% | "
                     f"{'-' if cf is None else f'{100 * cf:.1f}%'} |" if ours else f"| {name} | - | "
                     f"{100 * OCTO_BASE_PUBLISHED[task]:.1f}% | - |")
    lines += ["", "논문 값은 Google task에서 URDF 4종 전체 평균이고 우리 Octo는 원래 URDF만이라 Google 행은 근사 비교다."]
    return lines


def table_offline_scale(datasets):
    """Least-squares scalar s* with s*·pred ~= GT on the metre-space translation of the executed action."""
    import json as _json

    lines = ["| variant | embodiment | 최적 scale s* (오프라인) | 크기 비율 중앙값 |", "|---|---|---|---|"]
    stats = load_stats()
    for run in ["baseline_lang", "baseline_goal"]:
        for ds in datasets:
            p = RUNS / run / ds / "pred.npz"
            c = RES / "conventions" / f"{ds}.json"
            if not p.exists() or not c.exists():
                continue
            d = np.load(p)
            A = np.asarray(_json.loads(c.read_text())["A_action_to_m"])
            pred = unnormalize(d["pred_norm"], stats[ds]["action"])[:, 0, :3] @ A.T
            gt = d["gt"][:, 0, :3] @ A.T
            s_opt = float((pred * gt).sum() / (pred * pred).sum())
            mv = np.linalg.norm(gt, axis=1) > 0.002
            ratio = np.median(np.linalg.norm(pred[mv], axis=1) / np.linalg.norm(gt[mv], axis=1))
            lines.append(f"| {run} | {EMB_NAME.get(ds, ds)} | {s_opt:.2f} | {ratio:.2f} |")
    return lines


def fig_bars(rows, runs, datasets, path, title):
    keys = [("norm_l1", "normalized L1 (executed step)"), ("trans_err_mm_s", "translation error (mm/s)"),
            ("grip_acc", "gripper accuracy")]
    present = [r for r in runs if any((r, ds, "self") in rows for ds in datasets)]
    if not present:
        return False
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.8))
    width = 0.8 / max(1, len(datasets))
    for ax, (k, lab) in zip(axs, keys):
        for i, ds in enumerate(datasets):
            xs, ys, err = [], [], []
            for j, run in enumerate(present):
                r = rows.get((run, ds, "self"))
                if r is None:
                    continue
                v = r["metrics"][k]
                xs.append(j + i * width)
                ys.append(v["mean"])
                err.append([[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]])
            if xs:
                e = np.concatenate(err, axis=1) if err else None
                ax.bar(xs, ys, width, yerr=e, capsize=2, label=EMB_NAME.get(ds, ds))
        ax.set_xticks([j + width * (len(datasets) - 1) / 2 for j in range(len(present))])
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10)
    axs[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_horizon(rows, run, datasets, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ok = False
    for ds in datasets:
        r = rows.get((run, ds, "self"))
        if r:
            ax.plot(range(4), r["metrics"]["chunk_norm_l1_by_h"], "o-", label=EMB_NAME.get(ds, ds))
            ok = True
    if not ok:
        plt.close(fig)
        return False
    ax.set_xlabel("chunk step h (0 = executed)")
    ax.set_ylabel("normalized L1")
    ax.set_xticks(range(4))
    ax.set_title(f"{run}: error along the action chunk", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_trajectory(run, ds, path, stats):
    p = RUNS / run / ds / "pred.npz"
    if not p.exists():
        return False
    d = np.load(p)
    eps, counts = np.unique(d["episode"], return_counts=True)
    e = eps[np.argsort(counts)[len(counts) // 2]]  # a median-length episode
    sel = d["episode"] == e
    pred = unnormalize(d["pred_norm"][sel], stats[ds]["action"])[:, 0]
    gt = d["gt"][sel][:, 0]
    labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    fig, axs = plt.subplots(1, 7, figsize=(18, 2.6))
    for i, ax in enumerate(axs):
        ax.plot(gt[:, i], label="GT")
        ax.plot(pred[:, i], label="CrossFormer")
        ax.set_title(labels[i], fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{EMB_NAME.get(ds, ds)}: held-out episode {e} ({run}, executed step)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    sim = load_sim()
    stats = load_stats()
    datasets = [d for d in EMB_NAME if any(k[1] == d for k in rows)]
    offline_csv(rows) if rows else None

    md = ["# T10 CrossFormer baseline report", "",
          "자동 생성 파일 (`bash experiments/scripts/report.sh`). 해석 규칙은 `experiments/docs/PROTOCOL.md`.", "",
          "평균은 episode 단위, [ ]는 episode bootstrap 95% CI. 오프라인 지표는 실행되는 첫 action(chunk step 0) 기준.", ""]

    conv = RES / "conventions/CONVENTIONS.md"
    if conv.exists():
        md += ["## 0. Embodiment별 action 규약 (데이터에서 측정)", ""] + conv.read_text().splitlines()[2:] + [""]

    for title, runs in PHASE.items():
        md += [f"## {title}", ""] + table_variants(rows, runs, datasets) + [""]
    md += ["## Phase 2/3 paired differences (같은 episode끼리 뺀 값, episode bootstrap)", ""]
    md += table_paired(rows, PHASE["Phase 2 conditioning / history"] + PHASE["Phase 3 observation / rate probes"][1:],
                       datasets) + [""]
    md += ["## Execution-constraint violations (window 비율, 괄호 = 같은 검사를 GT action에 적용한 기준선)", ""]
    md += table_violations(rows, ["baseline_lang", "baseline_goal"], datasets) + [""]
    md += ["## Phase 3 action-unit probe: 어떤 통계로 unnormalize하는가 (baseline_goal 예측 재사용)", ""]
    md += table_stats_probe(rows, "baseline_goal", datasets) + [""]
    md += ["## 단위가 다른 raw MSE를 합치면: embodiment별 비중", ""]
    md += table_unit_dominance(rows, "baseline_goal", datasets) + [""]
    md += ["## Phase 4 closed-loop (SimplerEnv visual matching)", ""] + sim_tables(sim) + [""]
    md += ["## Phase 5 simulator validity: 같은 설치에서 Octo-Base가 논문 값을 재현하는가", ""] + table_octo_validity(sim) + [""]
    md += ["## Phase 6 viability screening: 출력 변환만 바꾼 closed-loop", ""] + table_screening(sim) + [""]
    md += ["## 오프라인에서 예측한 action scale (closed-loop sweep과 비교용)", ""] + table_offline_scale(datasets) + [""]

    figs = []
    if fig_bars(rows, PHASE["Phase 1 baseline"], datasets, OUT / "fig_phase1.png", "Phase 1: frozen CrossFormer"):
        figs.append("fig_phase1.png")
    if fig_bars(rows, PHASE["Phase 2 conditioning / history"], datasets, OUT / "fig_phase2.png",
                "Phase 2: conditioning and history"):
        figs.append("fig_phase2.png")
    if fig_bars(rows, PHASE["Phase 3 observation / rate probes"], datasets, OUT / "fig_phase3.png",
                "Phase 3: observation / control-rate probes"):
        figs.append("fig_phase3.png")
    if fig_horizon(rows, "baseline_goal", datasets, OUT / "fig_chunk_horizon.png"):
        figs.append("fig_chunk_horizon.png")
    for ds in datasets:
        if fig_trajectory("baseline_goal", ds, OUT / f"fig_traj_{ds}.png", stats):
            figs.append(f"fig_traj_{ds}.png")
    if figs:
        md += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + [""]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"-> {OUT / 'REPORT.md'} ({len(rows)} metric files, {len(sim)} closed-loop files, {len(figs)} figures)")


if __name__ == "__main__":
    main()
