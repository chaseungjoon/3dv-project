"""Summary table for the Hybrid-init diagnostic run (experiments/scripts/diag_hybrid.sh).

Compares the diagnostic run with the Phase 1 baseline, per start frame:
  start 0  = the standard evaluation (whole sequence, same scoring as every other run)
  start N  = every env begins at reference frame N (pose + kettle from the reference) and runs to the end.
A policy that survives from frame 45/60/90 has learned the lift/carry segment that the baseline never reached.

    python experiments/tools/diag_summary.py --diag_run experiments/runs/_diag/<run> \
        --base_run experiments/runs/default_s0_... --epoch 2000 --starts 0 45 60 90
"""
import argparse
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--diag_run', required=True)
    p.add_argument('--base_run', required=True)
    p.add_argument('--epoch', type=int, required=True, help='diagnostic budget (baseline compared at the same epoch)')
    p.add_argument('--base_final_epoch', type=int, default=6000)
    p.add_argument('--starts', type=int, nargs='+', default=[0, 45, 60, 90, 120])
    p.add_argument('--diag_eval', default='experiments/results/_diag/eval')
    p.add_argument('--base_eval', default='experiments/results/eval')
    p.add_argument('--out', default='experiments/results/_diag/SUMMARY.md')
    return p.parse_args()


def load(eval_dir, sub, run_name, epoch):
    path = os.path.join(eval_dir, sub, run_name, f'{run_name}_{epoch:08d}.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        res = json.load(f)
    # object error one control step after the start: large when the policy's first action knocks the kettle
    # away (seen for the baseline, which never trained from these states)
    npz = path[:-5] + '.npz'
    res['obj_err_first'] = float(np.load(npz)['obj_pos_err'][0].mean()) if os.path.isfile(npz) else None
    return res


def fmt(x, nd=2):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'


def row(label, res, start):
    if res is None:
        return f'| {label} | (평가 없음) | | | | | | |'
    eps = res['episodes']
    seq = eps[0]['seq_len']
    # official first failure (any termination) and task first failure (object > eps or fall), absolute frames
    off = [e['first_fail_frame'] if e['first_fail_frame'] >= 0 else seq - 1 for e in eps]
    task = [round(e['task_progress'] * (seq - 1)) + 1 if e['task_progress'] < 1 else seq - 1 for e in eps]
    span = max(1, (seq - 1) - start)
    surv_off = np.mean([(f - start) / span for f in off])
    surv_task = np.mean([(f - start) / span for f in task])
    a = res['aggregate']
    return (f"| {label} | {fmt(a['official_success'])} | {fmt(a['task_success'])} | {fmt(a['fell'])} | "
            f"{np.mean(off):.0f} ({surv_off:.0%}) | {np.mean(task):.0f} ({surv_task:.0%}) | "
            f"{fmt(res['obj_err_first'], 3)} | {fmt(a['obj_pos_err_final'], 3)} |")


def main():
    a = parse()
    os.chdir(REPO)
    diag = os.path.basename(os.path.normpath(a.diag_run))
    base = os.path.basename(os.path.normpath(a.base_run))
    models = [
        (f'baseline @ {a.epoch}', base, a.epoch),
        (f'baseline @ {a.base_final_epoch}', base, a.base_final_epoch),
        (f'**진단 Hybrid @ {a.epoch}**', diag, a.epoch),
    ]

    lines = [
        '# 진단 run 요약: 랜덤 시작 프레임(Hybrid) vs baseline',
        '',
        f'- 진단 run: `{a.diag_run}`  /  baseline: `{a.base_run}`',
        '- 채점은 모두 `eval_default.yaml`(baseline과 같은 기준), 결정적 행동, 물리 반복 env 8개 평균.',
        '- "시작 N" = 참조 프레임 N의 자세와 주전자 위치에서 시작해 끝(프레임 154)까지 진행.',
        '  참조 동작: 프레임 ~29 잡기, ~45~60 들어 올리기, 60~120 주전자를 든 채 옆으로 약 1.1 m 걷기.',
        '- 첫 실패 프레임 = 공식 실패 조건(넘어짐/몸/물체/손목/접촉)이 처음 선 프레임. 괄호 = 시작부터 끝까지 중 버틴 비율.',
        '- 과제 첫 실패 = 물체가 참조에서 0.10 m 넘게 벗어나거나 넘어진 첫 프레임 (손목 조건과 무관).',
        '- 시작 직후 물체 오차 = 시작 1스텝 뒤 물체 오차. 그 상태를 학습한 적 없는 정책(baseline)은 첫 행동으로 주전자를',
        '  쳐낸다 (시작 60, 90에서 0.10~0.15 m). 진단 정책은 0.01 m 미만이라 리셋 자체(물리)의 문제는 아니다.',
        '',
    ]
    for s in a.starts:
        sub = 'det' if s == 0 else f'det_start{s}'
        lines += [f'## 시작 프레임 {s}' + (' (표준 평가: 시퀀스 전체)' if s == 0 else ''), '',
                  '| 모델 | 공식 성공 | 과제 성공 | 넘어짐 | 첫 실패 프레임 (버틴 비율) | 과제 첫 실패 (버틴 비율) | 시작 직후 물체 오차 m | 마지막 물체 오차 m |',
                  '|---|---|---|---|---|---|---|---|']
        for label, run, ep in models:
            ev = a.base_eval if (s == 0 and run == base) else a.diag_eval
            lines.append(row(label, load(ev, sub, run, ep), s))
        lines.append('')

    # learning curve at start 0 (same epochs), stochastic success is the smoother signal
    lines += ['## 학습 곡선 (프레임 0 시작, 같은 epoch)', '',
              '| epoch | baseline 진행률 (결정적) | 진단 진행률 (결정적) | baseline 과제 진행률 (확률적) | 진단 과제 진행률 (확률적) | 진단 공식 성공 (확률적) |',
              '|---|---|---|---|---|---|']
    for ep in range(500, a.epoch + 1, 500):
        bd, dd = load(a.base_eval, 'det', base, ep), load(a.diag_eval, 'det', diag, ep)
        bs, ds = load(a.base_eval, 'stoch', base, ep), load(a.diag_eval, 'stoch', diag, ep)
        g = lambda r, k: r['aggregate'][k] if r else None
        lines.append(f"| {ep} | {fmt(g(bd, 'progress'), 3)} | {fmt(g(dd, 'progress'), 3)} | "
                     f"{fmt(g(bs, 'task_progress'), 3)} | {fmt(g(ds, 'task_progress'), 3)} | {fmt(g(ds, 'official_success'))} |")
    lines.append('')

    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
