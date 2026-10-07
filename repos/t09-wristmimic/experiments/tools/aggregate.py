"""Collect evaluation results and training logs into tables, figures and REPORT.md.

    python experiments/tools/aggregate.py                 # last checkpoint of every run
    python experiments/tools/aggregate.py --budget 3000   # every comparison at epoch 3000

Reads   experiments/results/eval/{det,stoch}/<run>/<ckpt>.json (+ .npz)  and  experiments/runs/<run>/
Writes  experiments/results/report/  (REPORT.md, csv tables, fig_*.png)
Uses only the standard library, numpy, matplotlib and tensorboard (no GPU).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Reference categorical palette (light mode), assigned by experiment group in a fixed order.
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
GROUP_COLOR = OrderedDict([('baseline', '#2a2a2a'), ('minimal', '#2a78d6'), ('rq1', '#eb6834'), ('rq2', '#1baf7a')])
CRITICAL = '#e34948'

# (key, Korean label for tables, English label for figures, format)
METRICS = [
    ('official_success', '공식 성공', 'official success', '{:.2f}'),
    ('task_success', '과제 성공', 'task success', '{:.2f}'),
    ('progress', '진행률', 'progress', '{:.3f}'),
    ('task_progress', '과제 진행률', 'task progress', '{:.3f}'),
    ('obj_pos_err_mean', '물체 위치 오차 m', 'object pos err (m)', '{:.4f}'),
    ('obj_rot_err_mean', '물체 회전 오차 rad', 'object rot err (rad)', '{:.3f}'),
    ('wrist_pos_err_window', '손목 위치 오차(창) m', 'wrist pos err in window (m)', '{:.4f}'),
    ('wrist_rot_err_window', '손목 회전 오차(창) rad', 'wrist rot err in window (rad)', '{:.3f}'),
    ('wrist_reset_frames', '손목 reset 조건 프레임', 'wrist-reset frames', '{:.1f}'),
    ('fell', '넘어짐', 'fall rate', '{:.2f}'),
    ('foot_slip_dist', '발 미끄러짐 m', 'foot slip (m)', '{:.3f}'),
    ('joint_limit_body', '관절 한계(몸)', 'joint-limit ratio (body)', '{:.4f}'),
    ('torso_tilt_excess_mean_deg', '몸통 기울기 초과 deg', 'torso tilt excess (deg)', '{:.2f}'),
    ('key_body_err_mean', '몸 추적 오차 m', 'body tracking err (m)', '{:.4f}'),
    ('contact_recall', '접촉 재현율', 'contact recall', '{:.2f}'),
]
CAUSES = ['fall', 'body', 'object', 'wrist_L', 'wrist_R', 'contact']


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_dir', default='experiments/results/eval')
    p.add_argument('--runs_dir', default='experiments/runs')
    p.add_argument('--out', default='experiments/results/report')
    p.add_argument('--budget', type=int, default=0, help='compare every run at this epoch')
    p.add_argument('--det_subdir', default='det')
    p.add_argument('--stoch_subdir', default='stoch')
    return p.parse_args()


# ------------------------------------------------------------------ loading
def load_variants():
    spec = yaml.safe_load(open(os.path.join(REPO, 'experiments/configs/variants.yaml')))
    return spec['variants']


def variant_of(meta):
    if meta.get('variant'):
        return meta['variant']
    m = re.match(r'(.+?)_s\d+_', meta.get('run_name', ''))
    return m.group(1) if m else meta.get('run_name', '?')


def load_evals(folder):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, '*', '*.json'))):
        r = json.load(open(path))
        meta = r['meta']
        rows.append(dict(path=path, run=meta['run_name'], variant=variant_of(meta), seed=meta.get('seed'),
                         epoch=meta.get('epoch'), meta=meta, agg=r['aggregate'], episodes=r['episodes']))
    return rows


def train_log_stats(run_dir):
    out = {}
    log = os.path.join(run_dir, 'train.log')
    if os.path.isfile(log):
        txt = open(log, errors='ignore').read()
        tot = [float(x) for x in re.findall(r'total time: ([\d.]+)', txt)]
        fps = [float(x) for x in re.findall(r'fps step: ([\d.]+)', txt)]
        ep = [int(x) for x in re.findall(r'epoch_num:(\d+)', txt)]
        if tot:
            steady = tot[3:] or tot
            out['sec_per_epoch'] = float(np.median(steady))
            out['env_steps_per_s'] = float(np.median(fps[3:] or fps))
        if ep:
            out['last_epoch'] = max(ep)
    meta_path = os.path.join(run_dir, 'run_meta.json')
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        out.update({k: meta.get(k) for k in ('variant', 'seed', 'epochs', 'num_envs', 'wall_s_total',
                                               'peak_gpu_mem_mib', 'exit_code', 'git', 'gpu')})
    return out


def tb_scalars(run_dir, tags):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    files = glob.glob(os.path.join(run_dir, 'summaries', 'events.*'))
    if not files:
        return {}
    acc = EventAccumulator(os.path.join(run_dir, 'summaries'), size_guidance={'scalars': 0})
    acc.Reload()
    have = set(acc.Tags().get('scalars', []))
    return {t: ([e.step for e in acc.Scalars(t)], [e.value for e in acc.Scalars(t)]) for t in tags if t in have}


# ------------------------------------------------------------------ helpers
def fmt(v, f='{:.3f}'):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '-'
    return f.format(v)


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


def pick_checkpoint(rows, budget):
    """One evaluation row per run: the budget epoch, or the run's highest epoch."""
    by_run = defaultdict(list)
    for r in rows:
        by_run[r['run']].append(r)
    chosen, missing = {}, []
    for run, rs in by_run.items():
        if budget:
            m = [r for r in rs if r['epoch'] == budget]
            if m:
                chosen[run] = m[0]
            else:
                missing.append(f"{run} (max epoch {max(r['epoch'] or 0 for r in rs)})")
        else:
            chosen[run] = max(rs, key=lambda r: r['epoch'] or 0)
    return chosen, missing


def smooth(y, k=3):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode='edge')
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def write_csv(path, rows, keys):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------------------------------------------ figures
def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False,
        'savefig.dpi': 160, 'savefig.bbox': 'tight', 'axes.axisbelow': True,
    })
    return plt


def fig_learning_curves(plt, det_rows, stoch_rows, variants, out):
    """Small multiples: one panel per variant, default runs drawn in gray as the reference."""
    curves = defaultdict(dict)  # variant -> run -> (epochs, values) for each metric
    for mode, rows in (('det', det_rows), ('stoch', stoch_rows)):
        by_run = defaultdict(list)
        for r in rows:
            by_run[(r['variant'], r['run'])].append(r)
        for (v, run), rs in by_run.items():
            rs.sort(key=lambda r: r['epoch'] or 0)
            curves[v].setdefault(run, {})[mode] = rs
    if not curves:
        return None
    panels = [('stoch', 'official_success', 'official success rate (32 stochastic rollouts)'),
              ('stoch', 'task_success', f'task success rate'),
              ('det', 'progress', 'progress (deterministic actions, 8 envs)'),
              ('stoch', 'obj_pos_err_mean', 'object pos err (m)')]
    names = [v for v in variants if v in curves] + [v for v in curves if v not in variants]
    fig, axes = plt.subplots(len(names), len(panels), figsize=(3.2 * len(panels), 2.1 * len(names)),
                             squeeze=False, sharex=True)
    ref_runs = curves.get('default', {})
    for i, v in enumerate(names):
        color = GROUP_COLOR.get(variants.get(v, {}).get('group', 'baseline'), INK)
        for j, (mode, key, title) in enumerate(panels):
            ax = axes[i][j]
            if v != 'default':
                for run, d in ref_runs.items():
                    if mode in d:
                        ax.plot([r['epoch'] for r in d[mode]], [r['agg'].get(key) for r in d[mode]],
                                color=MUTED, lw=1.2, alpha=0.7)
            for run, d in curves[v].items():
                if mode in d:
                    ax.plot([r['epoch'] for r in d[mode]], [r['agg'].get(key) for r in d[mode]],
                            color=color, marker='o', ms=3)
            if i == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(v, color=INK, fontsize=9)
            if i == len(names) - 1:
                ax.set_xlabel('epoch')
    fig.suptitle('Learning curves per variant (gray = default runs for reference)', color=INK, fontsize=10)
    path = os.path.join(out, 'fig_learning_curves.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_train_curves(plt, runs_dir, run_names, variants, out):
    tags = [('rewards0/iter', 'mean episode reward'), ('episode_lengths/iter', 'mean episode length (frames)'),
            ('wm/info/episode_progress', 'train episode progress'),
            ('wm/info/episode_progress_ratio_90', 'share of episodes reaching 90%')]
    data = {}
    for run in run_names:
        s = tb_scalars(os.path.join(runs_dir, run), [t for t, _ in tags])
        if s:
            data[run] = s
    if not data:
        return None
    fig, axes = plt.subplots(1, len(tags), figsize=(3.4 * len(tags), 2.8), squeeze=False)
    seen = set()
    for run, s in sorted(data.items()):
        v = variant_of({'run_name': run})
        group = variants.get(v, {}).get('group', 'baseline')
        color = GROUP_COLOR.get(group, INK)
        for j, (tag, title) in enumerate(tags):
            if tag in s:
                x, y = s[tag]
                label = group if group not in seen else None
                axes[0][j].plot(x, smooth(y, 25), color=color, lw=1.2, alpha=0.85, label=label if j == 0 else None)
                axes[0][j].set_title(title, fontsize=9, color=INK)
                axes[0][j].set_xlabel('epoch')
        seen.add(group)
    axes[0][0].legend(fontsize=8, loc='lower right')
    path = os.path.join(out, 'fig_train_curves.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_compare(plt, table, variants, out, mode):
    keys = [('official_success', 'official success'), ('task_success', 'task success'),
            ('progress', 'progress'), ('obj_pos_err_mean', 'object pos err (m)'),
            ('wrist_pos_err_window', 'wrist pos err in window (m)'), ('foot_slip_dist', 'foot slip (m)'),
            ('torso_tilt_excess_mean_deg', 'torso tilt excess (deg)'), ('joint_limit_body', 'joint-limit ratio (body)')]
    names = [v for v in variants if v in table] + [v for v in table if v not in variants]
    if not names:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    x = np.arange(len(names))
    for ax, (key, title) in zip(axes.flat, keys):
        means, errs, cols = [], [], []
        for v in names:
            vals = [r['agg'].get(key) for r in table[v]]
            m, s, n = mean_std(vals)
            means.append(m if m is not None else np.nan)
            errs.append(s if s is not None else 0)
            cols.append(GROUP_COLOR.get(variants.get(v, {}).get('group', 'baseline'), INK))
            for val in vals:
                if val is not None:
                    ax.plot(names.index(v), val, 'o', ms=4, color='white', mec=INK, mew=0.8, zorder=3)
        ax.bar(x, means, width=0.7, color=cols, edgecolor='white', linewidth=2, yerr=errs,
               error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        if key in ('official_success', 'task_success', 'progress'):
            ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
        ax.grid(axis='x', visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    fig.legend(handles, list(GROUP_COLOR), loc='upper right', ncol=4, fontsize=8)
    fig.suptitle(f'Variant comparison ({mode} eval; bar = mean over seeds, dots = seeds, '
                 'all scored with the default eval config)', color=INK, fontsize=10, x=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out, f'fig_compare_{mode}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_tradeoff(plt, table, variants, out):
    """Grasp/task outcome vs whole-body stability, one labeled point per variant (stochastic eval)."""
    pairs = [('foot_slip_dist', 'foot slip (m)  ->  less stable', 'task_success', 'task success rate'),
             ('torso_tilt_excess_mean_deg', 'torso tilt excess (deg)  ->  less stable', 'obj_pos_err_mean', 'object pos err (m)')]
    names = [v for v in variants if v in table]
    if not names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (kx, lx, ky, ly) in zip(axes, pairs):
        for v in names:
            mx, _, _ = mean_std([r['agg'].get(kx) for r in table[v]])
            my, _, _ = mean_std([r['agg'].get(ky) for r in table[v]])
            if mx is None or my is None:
                continue
            c = GROUP_COLOR.get(variants[v].get('group', 'baseline'), INK)
            ax.plot(mx, my, 'o', ms=8, color=c, mec='white', mew=2)
            ax.annotate(v, (mx, my), xytext=(5, 4), textcoords='offset points', fontsize=8, color=INK2)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, ms=7) for c in GROUP_COLOR.values()]
    axes[1].legend(handles, list(GROUP_COLOR), fontsize=8, loc='best')
    fig.suptitle('Grasp outcome vs whole-body stability per wrist setting (stochastic eval, mean over seeds)',
                 color=INK, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, 'fig_tradeoff.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_rq1(plt, table, out):
    """RQ1: sensitivity of each metric to the stage-2 position vs rotation threshold (x0.5, x1, x2)."""
    lines = {'position threshold (stage 2)': ['pos2_x0.5', 'default', 'pos2_x2'],
             'rotation threshold (stage 2)': ['rot2_x0.5', 'default', 'rot2_x2']}
    if not any(v in table for v in ('pos2_x0.5', 'pos2_x2', 'rot2_x0.5', 'rot2_x2')):
        return None
    keys = [('official_success', 'official success'), ('task_success', 'task success'),
            ('progress', 'progress'), ('foot_slip_dist', 'foot slip (m)')]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3))
    colors = ['#2a78d6', '#eb6834']
    for ax, (key, title) in zip(axes, keys):
        for (label, vs), c in zip(lines.items(), colors):
            xs, ys = [], []
            for xm, v in zip([0.5, 1, 2], vs):
                if v in table:
                    m, _, _ = mean_std([r['agg'].get(key) for r in table[v]])
                    if m is not None:
                        xs.append(xm)
                        ys.append(m)
            ax.plot(xs, ys, marker='o', ms=6, color=c, label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks([0.5, 1, 2])
        ax.set_xticklabels(['x0.5', 'x1', 'x2'])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('threshold multiplier')
    axes[0].legend(fontsize=8)
    fig.suptitle('RQ1: position vs rotation threshold sensitivity (stochastic eval)', color=INK, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, 'fig_rq1_sensitivity.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_series(plt, row, out, thresholds):
    """Per-frame view of one deterministic episode: wrist error vs the reset thresholds, object
    error vs eps, pelvis height, and where the first termination fired."""
    npz = row['path'][:-5] + '.npz'
    if not os.path.isfile(npz):
        return None
    s = np.load(npz)
    e = 0
    frames = s['frame'][:, e]
    ep = row['episodes'][e]
    hands = [h for h, n in enumerate('LR') if ep.get(f'wrist_{n}_active')] or [0, 1]
    fig, axes = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)

    def shade(ax, h):
        st = s['wrist_stage'][:, e, h]
        for k, alpha in ((1, 0.10), (2, 0.22), (3, 0.10)):
            on = frames[st == k]
            if on.size:
                ax.axvspan(on.min() - 0.5, on.max() + 0.5, color=MUTED, alpha=alpha, lw=0)

    ax = axes[0]
    for h in hands:
        shade(ax, h)
        ax.plot(frames, s['wrist_pos_err'][:, e, h], color='#2a78d6', label=f"{'LR'[h]} wrist")
        st = s['wrist_stage'][:, e, h]
        for k in (1, 2, 3):
            on = frames[st == k]
            if on.size:
                ax.hlines(thresholds['pos'][k - 1], on.min() - 0.5, on.max() + 0.5, color=INK2, lw=1, ls='--')
    ax.set_ylabel('wrist pos err (m)')
    ax.set_title('shaded = wrist window (dark = stage 2), dashed = reset threshold', fontsize=8, color=INK2)
    ax = axes[1]
    for h in hands:
        shade(ax, h)
        ax.plot(frames, s['wrist_rot_err'][:, e, h], color='#2a78d6')
        st = s['wrist_stage'][:, e, h]
        for k in (1, 2, 3):
            on = frames[st == k]
            if on.size:
                ax.hlines(thresholds['rot'][k - 1], on.min() - 0.5, on.max() + 0.5, color=INK2, lw=1, ls='--')
    ax.set_ylabel('wrist rot err (rad)')
    ax = axes[2]
    ax.plot(frames, s['obj_pos_err'][:, e], color='#eb6834')
    ax.axhline(row['meta'].get('task_eps_m', 0.1), color=INK2, lw=1, ls='--')
    ax.set_ylabel('object pos err (m)')
    ax = axes[3]
    ax.plot(frames, s['pelvis_h'][:, e], color='#1baf7a', label='pelvis height')
    ax.axhline(0.3, color=INK2, lw=1, ls='--')
    ax.set_ylabel('pelvis height (m)')
    ax.set_xlabel('frame (30 Hz)')
    ff = ep.get('first_fail_frame', -1)
    if ff is not None and ff >= 0:
        cause = ','.join(c for c in CAUSES if ep.get(f'first_cause_{c}')) or '?'
        for a in axes:
            a.axvline(ff, color=CRITICAL, lw=1.5)
        axes[0].annotate(f'first termination: {cause} @ {ff}', (ff, axes[0].get_ylim()[1]), xytext=(4, -12),
                         textcoords='offset points', fontsize=8, color=INK)
    fig.suptitle(f"{row['run']}  epoch {row['epoch']}  (deterministic actions, env 0)", color=INK, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, f"fig_series_{row['run']}_e{row['epoch']}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ report
def suggest_budget(stoch_rows, det_rows):
    """Phase 1 rule (PROTOCOL.md 5.2): the first epoch where the 3-checkpoint moving average of
    the stochastic task progress reaches 95% of its maximum."""
    rows = [r for r in stoch_rows if r['variant'] == 'default' and r['seed'] == 0] or \
           [r for r in det_rows if r['variant'] == 'default' and r['seed'] == 0]
    by_run = defaultdict(list)
    for r in rows:
        by_run[r['run']].append(r)
    if not by_run:
        return None, []
    run = max(by_run, key=lambda k: len(by_run[k]))
    rs = sorted(by_run[run], key=lambda r: r['epoch'] or 0)
    ep = [r['epoch'] for r in rs]
    y = smooth([r['agg'].get('task_progress', 0) for r in rs])
    top = float(np.max(y)) if len(y) else 0
    t = next((e for e, v in zip(ep, y) if v >= 0.95 * top), None) if top > 0 else None
    return t, list(zip(ep, [r['agg'] for r in rs], y))


def main():
    a = parse()
    os.chdir(REPO)
    variants = load_variants()
    os.makedirs(a.out, exist_ok=True)
    det = load_evals(os.path.join(a.eval_dir, a.det_subdir))
    stoch = load_evals(os.path.join(a.eval_dir, a.stoch_subdir))
    if not det and not stoch:
        raise SystemExit(f'평가 결과가 없습니다: {a.eval_dir}/{{{a.det_subdir},{a.stoch_subdir}}}')

    # every checkpoint, aggregated over envs -> learning-curve tables
    metric_keys = [k for k, *_ in METRICS] + [f'first_cause_{c}' for c in CAUSES]
    for mode, rows in (('det', det), ('stoch', stoch)):
        write_csv(os.path.join(a.out, f'checkpoints_{mode}.csv'),
                  [dict(run=r['run'], variant=r['variant'], seed=r['seed'], epoch=r['epoch'],
                        n=r['agg'].get('n_episodes'), **{k: r['agg'].get(k) for k in metric_keys}) for r in rows],
                  ['run', 'variant', 'seed', 'epoch', 'n'] + metric_keys)
        eps = []
        for r in rows:
            for i, e in enumerate(r['episodes']):
                eps.append(dict(run=r['run'], variant=r['variant'], seed=r['seed'], epoch=r['epoch'], env=i, **e))
        if eps:
            write_csv(os.path.join(a.out, f'episodes_{mode}.csv'), eps, list(eps[0].keys()))

    # one checkpoint per run (budget epoch or last) -> comparison tables
    sel = {}
    missing = []
    for mode, rows in (('det', det), ('stoch', stoch)):
        chosen, miss = pick_checkpoint(rows, a.budget)
        sel[mode] = chosen
        missing += [f'{mode}: {m}' for m in miss]
    tables = {}
    for mode in ('det', 'stoch'):
        t = defaultdict(list)
        for r in sel[mode].values():
            t[r['variant']].append(r)
        for v in t:
            t[v].sort(key=lambda r: (r['seed'] is None, r['seed']))
        tables[mode] = t

    runs = sorted({r['run'] for r in det + stoch})
    res = {run: train_log_stats(os.path.join(a.runs_dir, run)) for run in runs}
    write_csv(os.path.join(a.out, 'runs.csv'), [dict(run=k, **v) for k, v in res.items()],
              ['run', 'variant', 'seed', 'epochs', 'last_epoch', 'num_envs', 'sec_per_epoch', 'env_steps_per_s',
               'wall_s_total', 'peak_gpu_mem_mib', 'exit_code', 'git', 'gpu'])

    # ---------------- figures ----------------
    plt = setup_mpl()
    figs = OrderedDict()
    figs['학습 곡선 (평가)'] = fig_learning_curves(plt, det, stoch, variants, a.out)
    figs['학습 곡선 (학습 중 로그)'] = fig_train_curves(plt, a.runs_dir, runs, variants, a.out)
    figs['변형 비교 (확률적 평가)'] = fig_compare(plt, tables['stoch'], variants, a.out, 'stoch')
    figs['변형 비교 (결정적 평가)'] = fig_compare(plt, tables['det'], variants, a.out, 'det')
    figs['성공 vs 안정성'] = fig_tradeoff(plt, tables['stoch'], variants, a.out)
    figs['RQ1 위치/회전 임계값 민감도'] = fig_rq1(plt, tables['stoch'], a.out)
    eval_cfg = yaml.safe_load(open(os.path.join(REPO, 'experiments/configs/env/eval_default.yaml')))['env']
    thr = {'pos': [eval_cfg[f'wristPosResetThreshold{k}'] for k in (1, 2, 3)],
           'rot': [eval_cfg[f'wristRotResetThreshold{k}'] for k in (1, 2, 3)]}
    series_figs = [fig_series(plt, r, a.out, thr) for r in sorted(sel['det'].values(), key=lambda r: r['run'])]

    # ---------------- REPORT.md ----------------
    L = []
    L.append('# T09 WristMimic 실험 결과 (자동 생성)\n')
    L.append(f'- 생성: `experiments/tools/aggregate.py`  (평가 결과 {len(det)}개 결정적, {len(stoch)}개 확률적 체크포인트)')
    L.append(f"- 비교 기준 체크포인트: {'모든 run의 epoch ' + str(a.budget) + ' (학습 예산 T)' if a.budget else '각 run의 마지막 체크포인트 (예산 T 미지정: 비교가 같은 예산이 아닐 수 있음)'}")
    L.append('- 채점: 모든 변형을 `experiments/configs/env/eval_default.yaml`(기본 손목 임계값)로 평가. '
             '공식 성공 = 시퀀스 끝까지 어떤 termination도 없음. 과제 성공 = 물체 위치 오차가 항상 0.10 m 이하이고 넘어지지 않음 (손목 조건과 무관).')
    L.append('- 이 수치는 논문 재현이 아니라 **축소 설정(1024 env, PhysX 버퍼 축소, RTX 5070 12GB) baseline**이다. 시뮬레이터 성공률이며 실로봇 성공률이 아니다. 장면 1개, 장면당 정책 1개이므로 일반화를 주장하지 않는다.')
    if missing:
        L.append(f'- ⚠️ 예산 epoch 체크포인트가 없어 제외된 run: {", ".join(missing)}')
    L.append('')

    t, curve = suggest_budget(stoch, det)
    if curve:
        L.append('## 1. 학습 예산 T 후보 (Phase 1, default seed 0)\n')
        L.append('규칙(PROTOCOL.md 5.2): 확률적 평가의 과제 진행률을 체크포인트 3개 이동평균했을 때, 최댓값의 95%에 처음 도달한 epoch. '
                 '최종 T는 이 값과 시간 예산 중 작은 쪽으로 팀이 정한다.\n')
        L.append(f"**제안 T = {t if t else '판단 불가 (곡선이 아직 오르지 않음)'}**\n")
        L.append('| epoch | 공식 성공 | 과제 성공 | 진행률 | 과제 진행률 | 과제 진행률(이동평균) | 물체 위치 오차 m |')
        L.append('|---|---|---|---|---|---|---|')
        for e, g, sm in curve:
            L.append(f"| {e} | {fmt(g.get('official_success'), '{:.2f}')} | {fmt(g.get('task_success'), '{:.2f}')} | "
                     f"{fmt(g.get('progress'))} | {fmt(g.get('task_progress'))} | {fmt(sm)} | {fmt(g.get('obj_pos_err_mean'), '{:.4f}')} |")
        L.append('')

    def seed_table(mode, title):
        rows = tables[mode].get('default', [])
        if not rows:
            return
        L.append(f'## {title}\n')
        cols = METRICS
        L.append('| seed | epoch | ' + ' | '.join(c[1] for c in cols) + ' | 첫 실패 원인 |')
        L.append('|' + '---|' * (len(cols) + 3))
        for r in rows:
            causes = ', '.join(f"{c}:{r['agg'].get('first_cause_' + c):.2f}"
                               for c in CAUSES if r['agg'].get('first_cause_' + c)) or '-'
            L.append(f"| {r['seed']} | {r['epoch']} | " + ' | '.join(fmt(r['agg'].get(k), f) for k, _, _, f in cols) + f' | {causes} |')
        cells = []
        for k, _, _, f in cols:
            m, s, n = mean_std([r['agg'].get(k) for r in rows])
            cells.append(fmt(m, f) + (f' ± {fmt(s, f)}' if s is not None else ''))
        L.append(f'| **평균±표준편차 (n={len(rows)})** | | ' + ' | '.join(cells) + ' | |')
        L.append('')

    seed_table('det', '2. Baseline: 기본 설정, seed별 (결정적 행동, 물리 반복 env 8개의 평균)')
    seed_table('stoch', '3. Baseline: 기본 설정, seed별 (확률적 평가, env 32개의 평균)')

    def compare_table(mode, title):
        t = tables[mode]
        if not t:
            return
        L.append(f'## {title}\n')
        L.append('값 = seed 평균 (± 표준편차, seed 2개 이상일 때). n = seed 수.\n')
        cols = METRICS
        L.append('| 변형 | 그룹 | n | ' + ' | '.join(c[1] for c in cols) + ' |')
        L.append('|' + '---|' * (len(cols) + 3))
        names = [v for v in variants if v in t] + [v for v in t if v not in variants]
        for v in names:
            cells = []
            for k, _, _, f in cols:
                m, s, n = mean_std([r['agg'].get(k) for r in t[v]])
                cells.append(fmt(m, f) + (f' ± {fmt(s, f)}' if s is not None else ''))
            L.append(f"| `{v}` | {variants.get(v, {}).get('group', '-')} | {len(t[v])} | " + ' | '.join(cells) + ' |')
        L.append('')
        L.append('변형 설명: ' + '; '.join(f"`{v}` = {variants[v]['desc']}" for v in names if v in variants))
        L.append('')

    compare_table('stoch', '4. 변형 비교 (확률적 평가, env 32개) - 같은 예산, 같은 채점 기준')
    compare_table('det', '5. 변형 비교 (결정적 행동, env 8개)')

    # first-failure cause breakdown (stochastic episodes)
    if tables['stoch']:
        L.append('## 6. 첫 실패 원인 분포 (확률적 평가, 실패한 에피소드 중 비율)\n')
        L.append('한 프레임에 여러 조건이 동시에 걸릴 수 있어 합이 1을 넘을 수 있다. wrist_L/R = 손목 reset 조건(기본 임계값), '
                 'body = 몸 key body 0.35 m 초과, object = 물체 점 평균 0.5 m 초과, contact = 손 접촉 불일치 10프레임 초과, fall = 골반 높이 < 0.3 m.\n')
        L.append('| 변형 | 실패 에피소드 | ' + ' | '.join(CAUSES) + ' | 평균 첫 실패 프레임 |')
        L.append('|' + '---|' * (len(CAUSES) + 3))
        for v, rs in tables['stoch'].items():
            eps = [e for r in rs for e in r['episodes'] if not e['official_success']]
            if not eps:
                L.append(f'| `{v}` | 0 | ' + ' | '.join('-' for _ in CAUSES) + ' | - |')
                continue
            share = [np.mean([e[f'first_cause_{c}'] for e in eps]) for c in CAUSES]
            L.append(f'| `{v}` | {len(eps)} | ' + ' | '.join(f'{x:.2f}' for x in share)
                     + f" | {np.mean([e['first_fail_frame'] for e in eps]):.1f} |")
        L.append('')

    # hypothesis helpers
    st = tables['stoch']
    if any(v in st for v in ('pos2_x0.5', 'pos2_x2', 'rot2_x0.5', 'rot2_x2')):
        L.append('## 7. RQ1/H1 판정용: stage 2 임계값 x2 - x0.5 차이 (확률적 평가 seed 평균)\n')
        L.append('H1: 위치 임계값 쪽 변화량이 회전 임계값 쪽보다 크다. 위치 쪽 |Δ| ≤ 회전 쪽 |Δ| 이면 기각. (주 지표: 공식 성공률)\n')
        L.append('| 지표 | Δ위치 (pos2_x2 − pos2_x0.5) | Δ회전 (rot2_x2 − rot2_x0.5) |')
        L.append('|---|---|---|')
        for k, lab, _, f in METRICS[:4] + [METRICS[10], METRICS[12]]:
            def d(hi, lo):
                if hi in st and lo in st:
                    return mean_std([r['agg'].get(k) for r in st[hi]])[0] - mean_std([r['agg'].get(k) for r in st[lo]])[0]
                return None
            L.append(f"| {lab} | {fmt(d('pos2_x2', 'pos2_x0.5'), f)} | {fmt(d('rot2_x2', 'rot2_x0.5'), f)} |")
        L.append('')
    if 'gwp_30' in st and 'default' in st:
        L.append('## 8. RQ2/H2 판정용: gwp 70 → 30 (gwp_30 − default, 확률적 평가 seed 평균)\n')
        L.append('H2: 발 미끄러짐/몸통 기울기는 줄고(Δ<0) 물체 위치 오차는 커진다(Δ>0). 안정성이 개선되지 않거나 물체 오차가 커지지 않으면 기각.\n')
        L.append('| 지표 | Δ (gwp_30 − default) | Δ (gwp_140 − default) |')
        L.append('|---|---|---|')
        for k, lab, _, f in [METRICS[10], METRICS[12], METRICS[11], METRICS[4], METRICS[1], METRICS[0]]:
            base = mean_std([r['agg'].get(k) for r in st['default']])[0]
            vals = []
            for v in ('gwp_30', 'gwp_140'):
                m = mean_std([r['agg'].get(k) for r in st[v]])[0] if v in st else None
                vals.append(fmt(m - base, f) if (m is not None and base is not None) else '-')
            L.append(f'| {lab} | {vals[0]} | {vals[1]} |')
        L.append('')

    L.append('## 9. 자원 사용\n')
    L.append('| run | variant | seed | epochs | 마지막 epoch | env | epoch당 s | env-step/s | 총 시간 h | 최대 GPU MiB | exit |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for run, s in res.items():
        L.append(f"| {run} | {s.get('variant', '-')} | {s.get('seed', '-')} | {s.get('epochs', '-')} | {s.get('last_epoch', '-')} | "
                 f"{s.get('num_envs', '-')} | {fmt(s.get('sec_per_epoch'), '{:.2f}')} | {fmt(s.get('env_steps_per_s'), '{:.0f}')} | "
                 f"{fmt((s.get('wall_s_total') or float('nan')) / 3600, '{:.2f}')} | {s.get('peak_gpu_mem_mib') or '-'} | {s.get('exit_code', '-')} |")
    L.append('')

    L.append('## 10. 그림\n')
    for title, p in figs.items():
        if p:
            L.append(f'- {title}: `{os.path.basename(p)}`')
    for p in series_figs:
        if p:
            L.append(f'- 프레임별 오차 (실패 사례 분석): `{os.path.basename(p)}`')
    L.append('\n## 11. 파일\n')
    L.append('- `checkpoints_{det,stoch}.csv`: 체크포인트마다 env 평균 (학습 곡선용)')
    L.append('- `episodes_{det,stoch}.csv`: 에피소드 1개당 1행, 모든 지표')
    L.append('- `runs.csv`: run별 자원 사용')
    L.append('- 지표 정의: `experiments/docs/PROTOCOL.md` 4절')
    with open(os.path.join(a.out, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'[report] {os.path.join(a.out, "REPORT.md")}')
    for p in list(figs.values()) + series_figs:
        if p:
            print(f'[report] {p}')


if __name__ == '__main__':
    main()
