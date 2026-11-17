"""Failure analysis of one iteration (LOOP.md step 2). CPU only, seconds. Reads what src/evaluate.py wrote.

    python environment_iteration/src/analyze.py <iteration> [--epoch E]

Output: iterations/<it>/analysis/e<epoch>/
  summary.json     per cell: metrics (+ Wilson 95% CI of task_success), failure-mode counts, where/when each mode
                   breaks the task (frame + reference phase), event-frame statistics, median object path vs the
                   reference, the learning curve, and clip candidates (representative episodes per mode)
  episodes.csv     one row per evaluation episode
  REPORT.md        the same, readable (auto-generated: do not edit; write conclusions in ITERATION.md)
  object_path.png  median object height / distance along the path vs the reference, per cell
  failures.png     first-failure frame per mode over the reference phases
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import iterlib as L  # noqa: E402
import metrics as MX  # noqa: E402

KEY_FRAMES = [0, 30, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 147]
CURVE_KEYS = ["task_success", "paper_success", "lift_success", "task_progress", "held_frames_clip", "fell_in_clip"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


EVENT_KEYS = ("mode", "detail", "first_bad", "phase", "grasp_frame", "drop_frame", "off_desk_frame", "rot_off_frame",
              "floor_frame", "hold_streaks", "events")


def refresh_events(r, S, box):
    """Recompute the failure analysis from the stored trace with the CURRENT src/metrics.py, so a refined
    classifier never needs a GPU re-evaluation. The evaluate.py metrics are left as stored."""
    m = r["meta"]
    c = dict(n_clip=m["n_clip"], obj_z0=m["obj_z0"], ref_lift=m["ref_lift"], thr=m["hold_thr"],
             paper_z=m["paper_z"], paper_need=m["paper_need"], box=box)
    ph = MX.ref_phases(MX.clean_ref(S, c["n_clip"]), c["obj_z0"])
    m["ref_phases"] = ph
    for e, ep in enumerate(r["episodes"]):
        f = MX.failure_events(S, S["fall_t"], c, e, ph)
        ep.update({k: f[k] for k in EVENT_KEYS})
        ep["env"] = e
    r["aggregate"]["modes"] = {k: sum(e["mode"] == k for e in r["episodes"]) for k in MX.MODES}


def load_cell(eval_dir, cell, epoch, box=None):
    out = []
    for jf in sorted(glob.glob(os.path.join(eval_dir, cell, f"e{epoch}_s*.json"))):
        r = L.read_json(jf)
        nf = jf[:-5] + ".npz"
        S = dict(np.load(nf)) if os.path.isfile(nf) else None
        if S is not None and "fall_t" in S:
            refresh_events(r, S, box)
        out.append((r, S))
    return out


def q(vals, fn=np.median):
    vals = [v for v in vals if v is not None and v >= 0]
    return float(fn(vals)) if vals else None


def cell_summary(rows):
    eps = [e for r, _ in rows for e in r["episodes"]]
    n = len(eps)
    agg = MX.aggregate([{k: v for k, v in e.items() if not isinstance(v, (list, str, dict))} for e in eps])
    k = sum(e["task_success"] for e in eps)
    agg["task_success_ci95"] = wilson(k, n)
    modes = {}
    for m in MX.MODES:
        sel = [e for e in eps if e["mode"] == m]
        if not sel:
            continue
        ph = {}
        for e in sel:
            ph[e["phase"]] = ph.get(e["phase"], 0) + 1
        modes[m] = dict(n=len(sel), frac=len(sel) / n, first_bad_median=q([e["first_bad"] for e in sel]),
                        first_bad_p25=q([e["first_bad"] for e in sel], lambda v: np.percentile(v, 25)),
                        first_bad_p75=q([e["first_bad"] for e in sel], lambda v: np.percentile(v, 75)),
                        phases=ph, help=MX.MODE_HELP[m])
    ev = dict(
        grasp=dict(frac=np.mean([e["grasp_frame"] is not None for e in eps]),
                   median=q([e["grasp_frame"] for e in eps])),
        drop=dict(frac=np.mean([e["drop_frame"] is not None for e in eps]), median=q([e["drop_frame"] for e in eps])),
        off_desk=dict(frac=np.mean([e.get("off_desk_frame") is not None for e in eps]),
                      median=q([e.get("off_desk_frame") for e in eps])),
        rot_off=dict(frac=np.mean([e.get("rot_off_frame") is not None for e in eps]),
                     median=q([e.get("rot_off_frame") for e in eps])),
        floor=dict(frac=np.mean([e["floor_frame"] is not None for e in eps]),
                   median=q([e["floor_frame"] for e in eps])),
        fall=dict(frac=np.mean([e["fall_frame"] >= 0 for e in eps]), median=q([e["fall_frame"] for e in eps]),
                  in_clip_frac=float(np.mean([e["fell_in_clip"] for e in eps]))),
        track_lost=dict(median=q([e["first_bad"] for e in eps if e["mode"] != "success"])),
    )
    path = None
    traces = [S for _, S in rows if S is not None]
    if traces:
        meta = rows[0][0]["meta"]
        nc = meta["n_clip"]
        obj = np.concatenate([S["obj"][:nc] for S in traces], axis=1)      # (nc, N, 3)
        ref = np.concatenate([S["ref_obj"][:nc] for S in traces], axis=1)
        r0 = MX.clean_ref(traces[0], nc)
        start = r0[0]
        prog = np.linalg.norm((obj - start)[..., :2], axis=-1)
        refprog = np.linalg.norm((r0 - start)[:, :2], axis=-1)
        err = np.linalg.norm(obj - ref, axis=-1)
        path = dict(frames=[f for f in KEY_FRAMES if f < nc],
                    ref_z=[float(r0[f, 2]) for f in KEY_FRAMES if f < nc],
                    ref_dist=[float(refprog[f]) for f in KEY_FRAMES if f < nc],
                    z_median=[float(np.median(obj[f, :, 2])) for f in KEY_FRAMES if f < nc],
                    dist_median=[float(np.median(prog[f])) for f in KEY_FRAMES if f < nc],
                    err_median=[float(np.median(err[f])) for f in KEY_FRAMES if f < nc],
                    within_10cm=[float(np.mean(err[f] <= 0.10)) for f in KEY_FRAMES if f < nc])
    return dict(n=n, aggregate=agg, modes=modes, events=ev, path=path)


def clip_candidates(cells, rows_by_cell, per_mode=3):
    """Representative episodes per (cell, mode): closest to the mode's median first-failure frame."""
    out = []
    for cell, rows in rows_by_cell.items():
        eps = [(r["meta"]["seed"], e) for r, _ in rows for e in r["episodes"]]
        for m, info in cells[cell]["modes"].items():
            if m == "success":
                continue
            sel = sorted([(abs(e["first_bad"] - info["first_bad_median"]), s, e) for s, e in eps if e["mode"] == m],
                         key=lambda x: x[0])[:per_mode]
            for _, s, e in sel:
                out.append(dict(cell=cell, seed=s, env=e["env"] if "env" in e else None, mode=m,
                                first_bad=e["first_bad"], phase=e["phase"], grasp=e["grasp_frame"], rot_off=e.get("rot_off_frame"),
                                drop=e["drop_frame"], fall=e["fall_frame"], detail=e["detail"]))
    return out


def curve_table(eval_dir, box=None):
    rows = []
    for jf in glob.glob(os.path.join(eval_dir, "curve", "e*.json")):
        r = L.read_json(jf)
        nf = jf[:-5] + ".npz"
        if os.path.isfile(nf):
            S = dict(np.load(nf))
            if "fall_t" in S:
                refresh_events(r, S, box)
        g = r["aggregate"]
        rows.append(dict(epoch=r["meta"]["epoch"], **{k: g.get(k) for k in CURVE_KEYS},
                         modes={k: v for k, v in g.get("modes", {}).items() if v}))
    return sorted(rows, key=lambda x: x["epoch"])


def plots(out_dir, rows_by_cell, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [c for c in rows_by_cell if any(S is not None for _, S in rows_by_cell[c])]
    if not cells:
        return
    fig, ax = plt.subplots(2, len(cells), figsize=(6 * len(cells), 7), squeeze=False)
    for j, cell in enumerate(cells):
        traces = [S for _, S in rows_by_cell[cell] if S is not None]
        nc = rows_by_cell[cell][0][0]["meta"]["n_clip"]
        obj = np.concatenate([S["obj"][:nc] for S in traces], axis=1)
        r0 = MX.clean_ref(traces[0], nc)
        ph = rows_by_cell[cell][0][0]["meta"]["ref_phases"]
        prog = np.linalg.norm((obj - r0[0])[..., :2], axis=-1)
        refprog = np.linalg.norm((r0 - r0[0])[:, :2], axis=-1)
        t = np.arange(nc)
        for i, (y, ry, lab) in enumerate([(obj[..., 2], r0[:, 2], "object height (m)"),
                                          (prog, refprog, "horizontal distance from start (m)")]):
            a = ax[i, j]
            a.fill_between(t, np.percentile(y, 25, axis=1), np.percentile(y, 75, axis=1), alpha=0.3, label="IQR")
            a.plot(t, np.median(y, axis=1), label="median")
            a.plot(t, ry, "k--", label="reference")
            for name, (s, e) in ph.items():
                if s < nc:
                    a.axvline(s, color="gray", lw=0.6)
                    a.text(s + 1, a.get_ylim()[1] * 0.95 if i else ry.max(), name, fontsize=7, color="gray")
            a.set_xlabel("frame"); a.set_ylabel(lab); a.set_title(f"{cell} (n={obj.shape[1]})")
            a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "object_path.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, len(cells), figsize=(6 * len(cells), 3.5), squeeze=False)
    for j, cell in enumerate(cells):
        eps = [e for r, _ in rows_by_cell[cell] for e in r["episodes"] if e["mode"] != "success"]
        modes = [m for m in MX.MODES if any(e["mode"] == m for e in eps)]
        a = ax[0, j]
        for k, m in enumerate(modes):
            fb = [e["first_bad"] for e in eps if e["mode"] == m]
            a.scatter(fb, np.full(len(fb), k) + np.random.RandomState(k).uniform(-0.2, 0.2, len(fb)), s=12)
        a.set_yticks(range(len(modes))); a.set_yticklabels([f"{m} ({sum(e['mode'] == m for e in eps)})"
                                                            for m in modes])
        ph = rows_by_cell[cell][0][0]["meta"]["ref_phases"]
        for name, (s, e) in ph.items():
            a.axvline(s, color="gray", lw=0.6); a.text(s + 1, len(modes) - 0.5, name, fontsize=7, color="gray")
        a.set_xlabel("frame the task broke (first_bad)"); a.set_title(cell)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "failures.png"), dpi=110); plt.close(fig)


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def report(name, epoch, summary):
    pr = L.protocol()
    tgt = pr["target"]
    lines = [f"# {name} analysis at epoch {epoch} (auto-generated by src/analyze.py, do not edit)", "",
             f"Protocol v{pr['version']}: clean frame-{pr['eval']['start_frame']} starts, horizon "
             f"{pr['eval']['horizon']}. Target: {tgt['metric']} >= {100 * tgt['threshold']:.0f}% in cell "
             f"`{tgt['cell']}`. Frame t = control step after the reset (0..147); see src/metrics.py.", ""]
    lines += ["## Metrics", "", "| cell | n | task success (95% CI) | paper success | lift | progress | held frames | "
              "fell in clip |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for cell, s in summary["cells"].items():
        g = s["aggregate"]; lo, hi = g["task_success_ci95"]
        lines.append(f"| {cell} | {s['n']} | {fmt(g['task_success'], True)} ({100 * lo:.0f}-{100 * hi:.0f}%) | "
                     f"{fmt(g['paper_success'], True)} | {fmt(g['lift_success'], True)} | "
                     f"{fmt(g['task_progress'], True)} | {g['held_frames_clip']:.1f} | {fmt(g['fell_in_clip'], True)} |")
    for cell, s in summary["cells"].items():
        lines += ["", f"## Failure modes: `{cell}`", "", "| mode | episodes | first broken frame median [p25-p75] | "
                  "phase where it broke | meaning |", "|---|---:|---|---|---|"]
        for m, i in sorted(s["modes"].items(), key=lambda x: -x[1]["n"]):
            phs = ", ".join(f"{k} {v}" for k, v in sorted(i["phases"].items(), key=lambda x: -x[1]))
            lines.append(f"| {m} | {i['n']} ({100 * i['frac']:.0f}%) | {fmt(i['first_bad_median'])} "
                         f"[{fmt(i['first_bad_p25'])}-{fmt(i['first_bad_p75'])}] | {phs} | {i['help']} |")
        ev = s["events"]
        lines += ["", "| event | episodes with it | median frame |", "|---|---:|---:|"]
        for k in ("grasp", "off_desk", "rot_off", "drop", "floor", "fall"):
            lines.append(f"| {k} | {fmt(float(ev[k]['frac']), True)} | {fmt(ev[k]['median'])} |")
        if s["path"]:
            p = s["path"]
            lines += ["", "Median object path (all episodes of the cell):", "",
                      "| frame | " + " | ".join(map(str, p["frames"])) + " |",
                      "|---|" + "---:|" * len(p["frames"])]
            for lab, key, f in [("reference height (m)", "ref_z", "{:.2f}"), ("object height (m)", "z_median", "{:.2f}"),
                                ("reference distance (m)", "ref_dist", "{:.2f}"),
                                ("object distance (m)", "dist_median", "{:.2f}"),
                                ("error to reference (m)", "err_median", "{:.2f}"),
                                ("episodes within 10 cm", "within_10cm", "{:.0%}")]:
                lines.append(f"| {lab} | " + " | ".join(f.format(v) for v in p[key]) + " |")
    lines += ["", f"Reference phases (frames [start, end)): {summary['ref_phases']}", ""]
    if summary["curve"]:
        lines += ["## Learning curve (det, 8 envs, clean starts)", "", "| epoch | task | paper | lift | progress | "
                  "held | fell in clip | modes |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in summary["curve"]:
            lines.append(f"| {r['epoch']} | {fmt(r['task_success'], True)} | {fmt(r['paper_success'], True)} | "
                         f"{fmt(r['lift_success'], True)} | {fmt(r['task_progress'], True)} | "
                         f"{r['held_frames_clip']:.1f} | {fmt(r['fell_in_clip'], True)} | "
                         + ", ".join(f"{k} {v}" for k, v in r["modes"].items()) + " |")
    lines += ["", "## Clip candidates (representative episodes of each mode; videos re-run the policy, so pick by "
              "mode/frames in clips.json, not by env index)", "",
              "| cell | seed | env | mode | broke at | phase | grasp | rot off | drop | fall | detail |",
              "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|"]
    for c in summary["clip_candidates"]:
        lines.append(f"| {c['cell']} | {c['seed']} | {c['env']} | {c['mode']} | {c['first_bad']} | {c['phase']} | "
                     f"{fmt(c['grasp'])} | {fmt(c['rot_off'])} | {fmt(c['drop'])} | {fmt(c['fall'])} | {c['detail']} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--seed_run", type=int, default=None)
    a = p.parse_args()
    pr = L.protocol()
    epoch = a.epoch or pr["train"]["epochs"]
    P = L.paths(a.iteration, a.seed_run)
    rows_by_cell = {}
    box = MX.load_box(L.REPO, L.load(a.iteration).get("object_box", MX.BOOK_BOX))
    for cell in pr["eval"]["final"]:
        rows = load_cell(P["eval_dir"], cell["cell"], epoch, box)
        if rows:
            rows_by_cell[cell["cell"]] = rows
    if not rows_by_cell:
        sys.exit(f"[analyze] no final evaluation of {a.iteration} at epoch {epoch} "
                 f"(run scripts/eval.sh {a.iteration} first)")
    for c, rows in rows_by_cell.items():   # backfill env index (stored in episode order)
        for r, _ in rows:
            for i, e in enumerate(r["episodes"]):
                e.setdefault("env", i)
    cells = {c: cell_summary(rows) for c, rows in rows_by_cell.items()}
    first_meta = next(iter(rows_by_cell.values()))[0][0]["meta"]
    summary = dict(iteration=a.iteration, epoch=epoch, protocol_version=pr["version"], cells=cells,
                   ref_phases=first_meta["ref_phases"], curve=curve_table(P["eval_dir"], box),
                   clip_candidates=clip_candidates(cells, rows_by_cell),
                   seeds={c: sorted(r["meta"]["seed"] for r, _ in rows) for c, rows in rows_by_cell.items()})
    tgt = pr["target"]
    if tgt["cell"] in cells:
        v = cells[tgt["cell"]]["aggregate"][tgt["metric"]]
        summary["target"] = dict(tgt, value=v, reached=bool(v >= tgt["threshold"]))
    out = os.path.join(P["analysis_dir"], f"e{epoch}")
    os.makedirs(out, exist_ok=True)
    L.write_json(os.path.join(out, "summary.json"), summary)
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = None
        for c, rows in rows_by_cell.items():
            for r, _ in rows:
                for e in r["episodes"]:
                    row = dict(cell=c, seed=r["meta"]["seed"],
                               **{k: v for k, v in e.items() if k not in ("events", "hold_streaks")})
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
                    w.writerow(row)
    with open(os.path.join(out, "REPORT.md"), "w") as f:
        f.write(report(a.iteration, epoch, summary))
    plots(out, rows_by_cell, summary)
    t = summary.get("target")
    print(f"[analyze] {a.iteration} e{epoch}: " + ", ".join(
        f"{c} task {100 * s['aggregate']['task_success']:.1f}% (n={s['n']})" for c, s in cells.items())
        + (f" | target {'REACHED' if t['reached'] else 'not reached'}" if t else "") + f" -> {L.rel(out)}/")


if __name__ == "__main__":
    main()
