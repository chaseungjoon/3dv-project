"""Collect the PAPER vs FIX full-budget results into one folder (PAPER_FIX_FULL_BUDGET.md 7-11). No GPU, seconds.

Called by experiments/scripts/paper_budget/run_full.sh after every stage; safe to run by hand at any time and on
partial results (unfinished runs are reported as such):

    .venv/bin/python experiments/tools/paper_budget/summarize.py --config experiments/results_paper_budget/config.json

Reads    <runs_root>/<run>/{run_meta.json, train.log, nn/}  and  <results_root>/eval/<set>/<run>/*.json
Writes   <summary_dir>/
  summary.md          status, the pre-registered result table (section 11), the rule check, learning-curve notes
  status.json         machine-readable progress (epoch per run, done flags, eval counts)
  runs.csv            one row per training run: epoch reached, samples, wall hours, segments, failed segments, peak VRAM
  final_evals.csv     every evaluation of the final checkpoint: condition, training seed, eval set, eval seed, metrics
  primary.csv         per run, mean and SD over the pre-registered eval seeds (stoch and stoch+perturb)
  curve.csv           deterministic learning curve (8 env) at the pre-registered checkpoints
  train_reward.csv    mean training reward per epoch, parsed from train.log
  curves.png          learning curves (if matplotlib is available)
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics

EPOCH_RE = re.compile(r"epoch_num:(\d+) mean_rewards:\[([-\d.eE+na]+)\] fps step: ([\d.]+) fps total: ([\d.]+)")
NUM_RE = re.compile(r"_(\d{8})\.pth$")
METRICS = ["task_success", "task_progress", "obj_err_end", "obj_err_mean", "lift_success", "fell", "fell_in_clip",
           "paper_success", "held_frames_clip", "body_err_mean"]
LOWER_IS_BETTER = {"obj_err_end", "obj_err_mean", "fell", "fell_in_clip", "body_err_mean"}


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def fmt(x, pct=False, nd=3):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def fmt_ms(xs, pct=False):
    m, s = mean(xs), sd(xs)
    if m is None:
        return "–"
    return fmt(m, pct) + (" ± " + fmt(s, pct) if s is not None else "")


def sign(x, eps=1e-12):
    return 0 if x is None or abs(x) <= eps else (1 if x > 0 else -1)


class Run:
    def __init__(self, cfg, cond, seed):
        c = cfg["conditions"][cond]
        self.cond, self.seed, self.method = cond, seed, c["method"]
        self.name = f"{cfg['clip']}_{c['method']}_n{cfg['num_envs']}_s{seed}"
        self.dir = os.path.join(c["runs_dir"], self.name)
        self.results = os.path.join(c["results_dir"], "eval")
        self.target = cfg["epochs"]
        nums = [int(m.group(1)) for p in glob.glob(os.path.join(self.dir, "nn", "*.pth")) if (m := NUM_RE.search(p))]
        self.epoch = max(nums) if nums else 0
        self.done = self.epoch >= self.target
        self.meta = load_json(os.path.join(self.dir, "run_meta.json")) or {}

    def ev(self, eval_set, epoch):
        return load_json(os.path.join(self.results, eval_set, self.name, f"{self.name}_{epoch:08d}.json"))

    def eval_count(self):
        return len(glob.glob(os.path.join(self.results, "*", self.name, "*.json")))


def train_reward(run):
    rows, p = [], os.path.join(run.dir, "train.log")
    if os.path.isfile(p):
        seen = {}
        with open(p, errors="replace") as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if m:
                    try:
                        seen[int(m.group(1))] = (float(m.group(2)), float(m.group(4)))  # resumed epochs: last wins
                    except ValueError:
                        pass
        rows = [(e, r, fps) for e, (r, fps) in sorted(seen.items()) if e <= run.target]
    return rows


def rolling_stats(rows, w=100):
    if len(rows) < w:
        return None
    r = [x[1] for x in rows]
    roll = [sum(r[i - w:i]) / w for i in range(w, len(r) + 1)]
    best_i = max(range(len(roll)), key=roll.__getitem__)
    nan = sum(1 for x in r if math.isnan(x))
    return dict(final=roll[-1], best=roll[best_i], best_epoch=rows[best_i + w - 1][0], drop=roll[best_i] - roll[-1], nan=nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_json(a.config)
    out = cfg["summary_dir"]
    os.makedirs(out, exist_ok=True)
    E = cfg["epochs"]
    conds = list(cfg["conditions"])          # ["PAPER", "FIX"]
    seeds = cfg["seeds"]
    eval_seeds = cfg["eval_seeds"]
    runs = {(c, s): Run(cfg, c, s) for c in conds for s in seeds}
    order = [(o["cond"], o["seed"]) for o in cfg["order"]]

    # ---------------- runs.csv / status.json ----------------
    run_rows = []
    for (c, s) in order:
        r = runs[(c, s)]
        segs = r.meta.get("segments", [])
        failed = [g for g in segs if g.get("exit_code") not in (0, None)]
        run_rows.append(dict(condition=c, method=r.method, seed=s, run=r.name, epoch=r.epoch, target=E,
                             done=int(r.done), samples=r.epoch * r.meta.get("samples_per_epoch", cfg["num_envs"] * 32),
                             wall_h=round(r.meta.get("wall_s_total", 0) / 3600, 2), segments=len(segs),
                             failed_segments=len(failed), unfinished_segments=sum(1 for g in segs if "exit_code" not in g),
                             peak_gpu_mem_mib=r.meta.get("peak_gpu_mem_mib"), evals=r.eval_count()))
    write_csv(os.path.join(out, "runs.csv"), run_rows)
    all_done = all(r.done for r in runs.values())
    walls = [x["wall_h"] / x["epoch"] for x in run_rows if x["epoch"] >= 50 and x["wall_h"] > 0]
    sec_per_epoch = 3600 * mean(walls) if walls else None
    remaining_epochs = sum(E - x["epoch"] for x in run_rows)
    eta_h = remaining_epochs * sec_per_epoch / 3600 if sec_per_epoch else None
    with open(os.path.join(out, "status.json"), "w") as f:
        json.dump(dict(all_training_done=all_done, sec_per_epoch=sec_per_epoch, remaining_epochs=remaining_epochs,
                       eta_training_hours=eta_h, runs=run_rows), f, indent=1)

    # ---------------- final evaluations ----------------
    sets = [("stoch", 0, "stoch"), ("stoch_perturb", 0, "stoch_perturb")]
    for es in eval_seeds:
        sets += [("stoch", es, f"stoch_s{es}"), ("stoch_perturb", es, f"stoch_perturb_s{es}")]
    final_rows, prim = [], {}
    for (c, s) in order:
        r = runs[(c, s)]
        for mode, es, sub in sets:
            j = r.ev(sub, E)
            if j is None:
                continue
            agg = j["aggregate"]
            final_rows.append(dict(condition=c, seed=s, run=r.name, eval_set=mode, eval_seed=es, epoch=j["meta"].get("epoch"),
                                   n_episodes=agg.get("n_episodes"), **{m: agg.get(m) for m in METRICS}))
    write_csv(os.path.join(out, "final_evals.csv"), final_rows)
    for (c, s) in order:
        for mode in ("stoch", "stoch_perturb"):
            rows = [x for x in final_rows if x["condition"] == c and x["seed"] == s and x["eval_set"] == mode
                    and x["eval_seed"] in eval_seeds]
            prim[(c, s, mode)] = dict(n=len(rows), **{m: [x[m] for x in rows] for m in METRICS})
    prim_rows = []
    for (c, s, mode), d in prim.items():
        row = dict(condition=c, seed=s, eval_set=mode, n_eval_seeds=d["n"])
        for m in METRICS:
            row[m + "_mean"], row[m + "_sd"] = mean(d[m]), sd(d[m])
        prim_rows.append(row)
    write_csv(os.path.join(out, "primary.csv"), prim_rows)

    # ---------------- learning curves ----------------
    curve_rows, best_rows = [], []
    for (c, s) in order:
        r = runs[(c, s)]
        pts = []
        for e in cfg["curve_epochs"]:
            j = r.ev("det", e)
            if j is not None:
                agg = j["aggregate"]
                pts.append(dict(condition=c, seed=s, epoch=e, **{m: agg.get(m) for m in METRICS}))
        curve_rows += pts
        if pts:
            first = next((p["epoch"] for p in pts if (p["task_success"] or 0) > 0), None)
            best = max(pts, key=lambda p: ((p["task_success"] or 0), (p["task_progress"] or 0)))
            last = pts[-1]
            best_rows.append(dict(condition=c, seed=s, first_success_epoch=first, best_epoch=best["epoch"],
                                  best_success=best["task_success"], best_progress=best["task_progress"],
                                  final_epoch=last["epoch"], final_success=last["task_success"],
                                  final_progress=last["task_progress"]))
    write_csv(os.path.join(out, "curve.csv"), curve_rows)
    rew_rows, rew_stats = [], {}
    for (c, s) in order:
        rows = train_reward(runs[(c, s)])
        rew_rows += [dict(condition=c, seed=s, epoch=e, mean_reward=rw, fps_total=fps) for e, rw, fps in rows]
        rew_stats[(c, s)] = rolling_stats(rows)
    write_csv(os.path.join(out, "train_reward.csv"), rew_rows)
    plot(out, conds, seeds, curve_rows, rew_rows)

    # ---------------- summary.md ----------------
    L = [f"# PAPER vs FIX full budget ({cfg['clip']}, {E} epochs = {E * cfg['num_envs'] * 32:,} samples/run)", "",
         "Plan: `PAPER_FIX_FULL_BUDGET.md` (project root). Generated by `experiments/tools/paper_budget/summarize.py`; "
         "regenerated after every stage of `run_full.sh`.", ""]
    scope = cfg.get("scope", "plan")
    L += [f"Last launch: `--{scope}` (runs {', '.join(cfg.get('scope_runs', []))}). "
          "Scopes: `--s0` = PAPER s0 + FIX s0 (pilot), `--s0s1` = + FIX s1, PAPER s1, `--plan` = all 6 (pre-registered).", ""]
    if scope != "plan":
        L += [f"**Note:** with `--{scope}` the result is a reduced-seed run, not the pre-registered 3-seed design; "
              "the section-11 rules need 3 training seeds per condition, so the verdict below stays INCOMPLETE.", ""]
    L += ["## 1. Status", "", "|order|condition|seed|run|epoch|done|wall h|segments (failed)|peak VRAM MiB|eval files|",
          "|---:|---|---:|---|---:|:-:|---:|---|---:|---:|"]
    for i, x in enumerate(run_rows, 1):
        L.append(f"|{i}|{x['condition']}|{x['seed']}|`{x['run']}`|{x['epoch']:,}/{E:,}|{'✓' if x['done'] else ''}|"
                 f"{x['wall_h']}|{x['segments']} ({x['failed_segments']})|{x['peak_gpu_mem_mib'] or '–'}|{x['evals']}|")
    L.append("")
    if sec_per_epoch:
        L.append(f"Measured {sec_per_epoch:.2f} s/epoch. Remaining {remaining_epochs:,} epochs ≈ {eta_h:.1f} h of training.")
    L += ["", "## 2. Pre-registered result table (final checkpoint, epoch %d)" % E, "",
          f"Values per training seed are the mean over eval seeds {', '.join(map(str, eval_seeds))} (32 envs each). "
          "Bottom row: mean ± SD over the training seeds (the independent unit).", "",
          "|training seed|PAPER success|FIX success|PAPER perturb|FIX perturb|PAPER goal progress|FIX goal progress|"
          "PAPER final obj err (m)|FIX final obj err (m)|",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    P, F = conds[0], conds[1]

    def pm(c, s, mode, m):
        return mean(prim[(c, s, mode)][m])
    cols = [("stoch", "task_success", True), ("stoch_perturb", "task_success", True),
            ("stoch", "task_progress", True), ("stoch", "obj_err_end", False)]
    for s in seeds:
        cells = []
        for mode, m, pct in cols:
            cells += [fmt(pm(P, s, mode, m), pct), fmt(pm(F, s, mode, m), pct)]
        L.append(f"|{s}|" + "|".join(cells) + "|")
    cells = []
    for mode, m, pct in cols:
        cells += [fmt_ms([pm(P, s, mode, m) for s in seeds], pct), fmt_ms([pm(F, s, mode, m) for s in seeds], pct)]
    L.append("|mean ± SD|" + "|".join(cells) + "|")
    L += ["", "Secondary (stoch, mean over eval seeds; mean ± SD over training seeds):", "",
          "|metric|PAPER|FIX|PAPER perturb|FIX perturb|", "|---|---:|---:|---:|---:|"]
    for m in ["lift_success", "fell", "fell_in_clip", "paper_success", "obj_err_mean", "body_err_mean", "held_frames_clip"]:
        pct = m in ("lift_success", "fell", "fell_in_clip", "paper_success")
        L.append(f"|{m}|" + "|".join(fmt_ms([pm(c, s, mode, m) for s in seeds], pct)
                                    for mode in ("stoch", "stoch_perturb") for c in (P, F)) + "|")

    # ---------------- rule check (section 11) ----------------
    L += ["", "## 3. Rule check (section 11, mechanical)", ""]
    complete = all_done and all(prim[(c, s, mode)]["n"] == len(eval_seeds)
                                for c in conds for s in seeds for mode in ("stoch", "stoch_perturb"))
    if not complete:
        L += ["**INCOMPLETE**: not every run has its final checkpoint evaluated with every eval seed. "
              "Re-run `run_full.sh`.", ""]
    verdict_lines, verdict = rule_check(prim, P, F, seeds)
    L += verdict_lines
    L += ["", f"**Mechanical verdict: {verdict if complete else 'INCOMPLETE (' + verdict + ' on partial data)'}**", "",
          "The rule check is an aid, not the conclusion. Still check by hand: is the success gain explained by fewer "
          "falls alone? Does the same direction show at the best checkpoint *and* at the final one? "
          "Claims stay at the level of the whole fix package (not mass / collision / phase separately)."]

    # ---------------- learning stability ----------------
    L += ["", "## 4. Learning curve and stability (secondary / exploratory)", "",
          "Deterministic 8-env curve at the pre-registered checkpoints (`curve.csv`); training reward from train.log, "
          "100-epoch rolling mean (`train_reward.csv`). Best checkpoint is exploratory only.", "",
          "|condition|seed|first det success epoch|best det epoch (success / progress)|final det success / progress|"
          "train reward final|best (epoch)|drop best→final|NaN epochs|",
          "|---|---:|---:|---|---|---:|---|---:|---:|"]
    br = {(b["condition"], b["seed"]): b for b in best_rows}
    for (c, s) in order:
        b, rs = br.get((c, s)), rew_stats.get((c, s))
        L.append(f"|{c}|{s}|{(b or {}).get('first_success_epoch') or '–'}|"
                 + (f"{b['best_epoch']} ({fmt(b['best_success'], True)} / {fmt(b['best_progress'], True)})|"
                    f"{fmt(b['final_success'], True)} / {fmt(b['final_progress'], True)}|" if b else "–|–|")
                 + (f"{rs['final']:.2f}|{rs['best']:.2f} ({rs['best_epoch']})|{rs['drop']:.2f}|{rs['nan']}|" if rs else "–|–|–|–|"))
    L += ["", "Figure: `curves.png`. Raw per-episode data: `<results_dir>/eval/<set>/<run>/*.json|npz`.", "",
          "## 5. Files", "", "|what|where|", "|---|---|"]
    for c in conds:
        L.append(f"|{c} checkpoints / logs|`{cfg['conditions'][c]['runs_dir']}/<run>/` (nn/, train.log, gpu.csv, run_meta.json)|")
        L.append(f"|{c} evaluations|`{cfg['conditions'][c]['results_dir']}/eval/<set>/<run>/`|")
    L += [f"|provenance (hashes, git, invocations)|`{out}/manifest/`|",
          f"|final checkpoint epoch/frame/SHA-256|`{out}/checkpoints.csv`|",
          f"|orchestrator logs (every launch)|`{out}/logs/`|"]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[summary] {out}/summary.md  (training done: {sum(r.done for r in runs.values())}/{len(runs)} runs, "
          f"{len(final_rows)} final evals, {len(curve_rows)} curve points)")


def rule_check(prim, P, F, seeds):
    L = []

    def per_seed(mode, m):
        out = []
        for s in seeds:
            p, f = mean(prim[(P, s, mode)][m]), mean(prim[(F, s, mode)][m])
            out.append(None if p is None or f is None else (f - p) * (-1 if m in LOWER_IS_BETTER else 1))
        return out  # > 0 means FIX better
    d_succ = per_seed("stoch", "task_success")
    n_fix = sum(1 for d in d_succ if d is not None and d > 0)
    n_pap = sum(1 for d in d_succ if d is not None and d < 0)
    direction = sign(mean(d_succ))
    dir_txt = {1: "FIX higher", -1: "PAPER higher", 0: "tie"}[direction]
    checks = []
    c1 = (n_fix >= 2) if direction > 0 else (n_pap >= 2) if direction < 0 else False
    checks.append((c1, f"Per-seed task success (stoch): FIX better in {n_fix}/{len(seeds)}, PAPER better in {n_pap}/{len(seeds)} "
                       f"(need ≥2 in the direction of the mean, which is: {dir_txt})"))
    d_prog, d_err = mean(per_seed("stoch", "task_progress")), mean(per_seed("stoch", "obj_err_end"))
    c2 = direction != 0 and sign(d_prog) == direction and sign(d_err) in (direction, 0)
    checks.append((c2, f"Goal progress and final object error point the same way as success "
                       f"(Δprogress FIX−PAPER = {fmt(d_prog)}, error improvement FIX vs PAPER = {fmt(d_err)} m)"))
    d_pert = mean(per_seed("stoch_perturb", "task_success"))
    c3 = direction != 0 and sign(d_pert) == direction
    checks.append((c3, f"Direction holds under object perturbation (Δperturb success = {fmt(d_pert, True)})"))
    var = [sd(prim[(c, s, "stoch")]["task_success"]) for c in (P, F) for s in seeds]
    var = [v for v in var if v is not None]
    pooled = math.sqrt(sum(v * v for v in var) / len(var)) if var else None
    dm = mean(d_succ)
    c4 = pooled is not None and dm is not None and abs(dm) > 2 * pooled
    checks.append((c4, f"Condition difference exceeds repeat-evaluation noise: |Δsuccess| = {fmt(abs(dm) if dm is not None else None, True)} "
                       f"vs 2 × pooled eval-seed SD = {fmt(2 * pooled if pooled is not None else None, True)}"))
    d_fall = mean(per_seed("stoch", "fell_in_clip"))
    L += [f"- [{'x' if ok else ' '}] {txt}" for ok, txt in checks]
    L.append(f"- [ ] (manual) Success change not explained by falls alone: fall-in-clip improvement FIX vs PAPER = {fmt(d_fall, True)}")
    big_one = sum(1 for d in d_succ if d is not None and abs(d) > 0) == 1
    if big_one:
        L.append("- Note: only one training seed shows any success difference (→ INCONCLUSIVE rule).")
    if all(ok for ok, _ in checks) and not big_one:
        verdict = "FIX SUPPORTED (fix package better at equal budget)" if direction > 0 else "FIX NOT BETTER (PAPER higher at equal budget)"
    else:
        verdict = "INCONCLUSIVE"
    return L, verdict


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["empty"])
        w.writeheader()
        w.writerows(rows)


def plot(out, conds, seeds, curve_rows, rew_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not curve_rows and not rew_rows:
        return
    colors = {conds[0]: "#4C72B0", conds[1]: "#DD8452"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for c in conds:
        for s in seeds:
            pts = [r for r in curve_rows if r["condition"] == c and r["seed"] == s]
            if pts:
                ax[0].plot([p["epoch"] for p in pts], [p["task_success"] for p in pts], "-o", ms=3, color=colors[c],
                           alpha=0.8, label=f"{c} s{s}")
                ax[1].plot([p["epoch"] for p in pts], [p["task_progress"] for p in pts], "-o", ms=3, color=colors[c], alpha=0.8)
            rw = [r for r in rew_rows if r["condition"] == c and r["seed"] == s]
            if len(rw) > 100:
                y = [r["mean_reward"] for r in rw]
                k = 100
                sm = [sum(y[i - k:i]) / k for i in range(k, len(y) + 1)]
                ax[2].plot([r["epoch"] for r in rw[k - 1:]], sm, color=colors[c], alpha=0.8, lw=1)
    for a_, t in zip(ax, ["det task_success (8 env)", "det task_progress (8 env)", "train mean reward (100-epoch mean)"]):
        a_.set_title(t); a_.set_xlabel("epoch"); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
