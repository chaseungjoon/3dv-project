"""Evaluate WristMimic checkpoints with the fixed T09 protocol (experiments/docs/PROTOCOL.md).

Builds the environment once and evaluates many checkpoints in one process, using the same
policy/env path as `run.py --test --test_no_reset` (one episode = the whole reference sequence,
no reset on failure, failure flag sticky). Every checkpoint is scored with the same eval config
(configs/env/eval_default.yaml = default wrist thresholds), whatever config it was trained with.

    # deterministic actions, 8 envs, every saved checkpoint of a run -> learning curve
    python experiments/tools/evaluate.py --run_dir experiments/runs/<run> --which all
    # stochastic actions, 32 envs, final checkpoint only
    python experiments/tools/evaluate.py --run_dir experiments/runs/<run> --mode stoch

Output: experiments/results/eval/<mode>/<run_name>/<ckpt>.json (+ .npz per-frame series).

GPU PhysX is not bit-reproducible: the same checkpoint with deterministic actions diverges at the
1e-5 m level after a few frames, between processes and between envs of one sim (checked
2026-10-04, PROTOCOL.md 4.3). So "det" runs 8 envs = 8 physics repeats of the deterministic
policy; the official single-env run.py --test result is one sample of that distribution.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULTS = dict(
    cfg_env='experiments/configs/env/eval_default.yaml',
    cfg_train='intermimic/data/cfg/train/rlg/parahome.yaml',
    motion_file='InterAct/Parahome/s110_0_kettle_table2desk',
    robot_type='sim_human/s110_ROM.xml',
    out_dir='experiments/results/eval',
)
CKPT_EPOCH_RE = re.compile(r'_(\d{8})\.pth$')


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', nargs='*', default=[], help='training run folder(s) (contain nn/)')
    p.add_argument('--checkpoints', nargs='*', default=[], help='explicit checkpoint files')
    p.add_argument('--which', default='final', choices=['final', 'all', 'latest'],
                   help='final = highest-numbered checkpoint; all = every numbered checkpoint')
    p.add_argument('--epochs', default='', help='comma-separated epochs to evaluate (overrides --which)')
    p.add_argument('--mode', default='det', choices=['det', 'stoch'])
    p.add_argument('--num_envs', type=int, default=0, help='default: det 8, stoch 32')
    p.add_argument('--seed', type=int, default=0, help='eval seed (only matters for --mode stoch)')
    p.add_argument('--cfg_env', default=DEFAULTS['cfg_env'])
    p.add_argument('--cfg_train', default=DEFAULTS['cfg_train'])
    p.add_argument('--motion_file', default=DEFAULTS['motion_file'])
    p.add_argument('--robot_type', default=DEFAULTS['robot_type'])
    p.add_argument('--out_dir', default=DEFAULTS['out_dir'])
    p.add_argument('--tag', default='', help='optional suffix for the output folder (e.g. pipeline)')
    p.add_argument('--start_frame', type=int, default=0,
                   help='diagnostic: start every env at this reference frame instead of 0 '
                        '(default tag becomes start<N>, so results never mix with the standard evaluation)')
    p.add_argument('--force', action='store_true', help='re-evaluate even if the result file exists')
    p.add_argument('--no_series', action='store_true', help='do not save the per-frame .npz')
    p.add_argument('--viewer', action='store_true', help='open the Isaac Gym viewer (needs a display)')
    p.add_argument('--image_dir', default='', help='with --viewer: save every frame as PNG here')
    return p.parse_args()


def ckpt_epoch(path):
    m = CKPT_EPOCH_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def select_checkpoints(run_dir, which, epochs):
    nn_dir = os.path.join(run_dir, 'nn')
    numbered = sorted((ckpt_epoch(p), p) for p in glob.glob(os.path.join(nn_dir, '*.pth')) if ckpt_epoch(p) is not None)
    if epochs:
        wanted = {int(e) for e in epochs.split(',') if e}
        return [p for e, p in numbered if e in wanted]
    if which == 'all':
        return [p for _, p in numbered]
    if which == 'latest':
        latest = glob.glob(os.path.join(nn_dir, '*_latest.pth'))
        return latest[:1]
    return [numbered[-1][1]] if numbered else []


def run_meta(run_dir):
    """variant / seed / budget of a training run (written by experiments/scripts/train.sh)."""
    meta = {'run_dir': os.path.relpath(run_dir, REPO), 'run_name': os.path.basename(os.path.normpath(run_dir))}
    path = os.path.join(run_dir, 'run_meta.json')
    if os.path.isfile(path):
        with open(path) as f:
            meta.update(json.load(f))
    return meta


def sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def git_commit():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(['git', 'diff', '--quiet', 'HEAD'], cwd=REPO, stderr=subprocess.DEVNULL) != 0
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def main():
    a = parse()
    os.chdir(REPO)  # configs, assets and motion paths are repo-relative
    sys.path.insert(0, os.path.join(REPO, 'intermimic'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault('WANDB_MODE', 'disabled')

    jobs = []  # (meta, checkpoint)
    for rd in a.run_dir:
        rd = os.path.abspath(rd)
        cks = select_checkpoints(rd, a.which, a.epochs)
        if not cks:
            print(f'[eval] no checkpoints in {rd}/nn, skipped')
        jobs += [(run_meta(rd), c) for c in cks]
    for c in a.checkpoints:
        c = os.path.abspath(c)
        rd = os.path.dirname(os.path.dirname(c))
        jobs.append((run_meta(rd), c))
    if not jobs:
        sys.exit('[eval] nothing to evaluate')

    num_envs = a.num_envs or (8 if a.mode == 'det' else 32)
    if a.start_frame and not a.tag:
        a.tag = f'start{a.start_frame}'
    if a.viewer:
        num_envs = 1
    out_root = os.path.join(a.out_dir, a.mode + (f'_{a.tag}' if a.tag else ''))

    def out_path(meta, ckpt):
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        return os.path.join(out_root, meta['run_name'], stem + '.json')

    if not a.force:
        todo = [(m, c) for m, c in jobs if not os.path.isfile(out_path(m, c))]
        if len(todo) < len(jobs):
            print(f'[eval] {len(jobs) - len(todo)} checkpoint(s) already evaluated, skipped (use --force to redo)')
        jobs = todo
    if not jobs:
        print('[eval] all done')
        return

    # ---- build args exactly like run.py --test --test_no_reset would get them ----
    sys.argv = [os.path.join(REPO, 'intermimic', 'run.py'),
                '--task', 'InterMimic_MULTI_OBJ',
                '--cfg_env', a.cfg_env, '--cfg_train', a.cfg_train,
                '--test', '--test_no_reset',
                '--checkpoint', jobs[0][1],
                '--num_envs', str(num_envs),
                '--motion_file', a.motion_file, '--robot_type', a.robot_type,
                '--num_position_iterations', '20', '--num_velocity_iterations', '0',
                '--seed', str(a.seed)]
    if not a.viewer:
        sys.argv.append('--headless')

    from isaacgym import gymapi  # noqa: F401  (must be imported before torch)
    import torch
    import numpy as np
    from easydict import EasyDict
    import wandb
    import run as R
    from metrics import EpisodeRecorder, TASK_EPS_M

    args = R.get_args()
    cfg, cfg_train, _ = R.load_cfg(args)
    cfg = EasyDict(cfg)
    cfg_train['params']['seed'] = R.set_seed(cfg_train['params'].get('seed', -1), False)
    cfg['env']['motion_file'] = a.motion_file
    cfg['env']['robotType'] = a.robot_type
    cfg['env']['testNoReset'] = True
    cfg['env']['startFrame'] = a.start_frame
    if a.viewer and a.image_dir:
        os.makedirs(a.image_dir, exist_ok=True)
        os.environ['INTERMIMIC_IMAGE_DIR'] = os.path.abspath(a.image_dir)
        cfg['env']['saveImages'] = True
    player_cfg = cfg_train['params']['config'].setdefault('player', {})
    player_cfg['determenistic'] = (a.mode == 'det')   # rl_games spelling
    R.args, R.cfg, R.cfg_train = args, cfg, cfg_train
    wandb.init(mode='disabled')

    runner = R.build_alg_runner(R.RLGPUAlgoObserver())
    runner.load(cfg_train)
    runner.reset()
    player = runner.create_player()
    task = player.env.task
    deterministic = (a.mode == 'det')
    print(f'[eval] env ready: num_envs={num_envs} mode={a.mode} cfg_env={a.cfg_env}')

    commit = git_commit()
    cfg_sha = sha1(a.cfg_env)
    for k, (meta, ckpt) in enumerate(jobs):
        t0 = time.time()
        player.restore(ckpt)
        state = torch.load(ckpt, map_location='cpu')
        epoch = state.get('epoch', ckpt_epoch(ckpt))
        frame = state.get('frame', None)
        del state
        if a.mode == 'stoch':
            torch.manual_seed(a.seed)

        # Full reset. A fresh run.py --test process sees an all-zero observation on its first
        # step (obs_buf is only computed after a physics step); zeroing it here keeps every
        # checkpoint's rollout independent of what was evaluated before it in this process.
        obs = player.env_reset()
        task.obs_buf.zero_()
        if hasattr(task, 'weighting_obs_buf'):
            task.weighting_obs_buf.zero_()
        obs = player.env_reset([])
        player.get_batch_size(obs['obs'], 1)

        rec = EpisodeRecorder(task)
        max_steps = int(task.max_episode_length.max().item()) + 5
        for _ in range(max_steps):
            if a.viewer:
                p = task._humanoid_root_states[0, 0:3].cpu().numpy()
                task.gym.viewer_camera_look_at(task.viewer, None, gymapi.Vec3(p[0] + 0.9, p[1] - 1.9, 1.45),
                                               gymapi.Vec3(p[0], p[1], 0.9))
            action = player.get_action(obs, deterministic)
            obs, reward, done, info = player.env_step(player.env, action)
            rec.step(info, reward)
            if done.any():
                if not done.all():
                    raise RuntimeError('envs finished at different frames; expected Start init + test_no_reset')
                break
        episodes, series = rec.summarize()

        # aggregate over envs (deterministic: 1 env; stochastic: num_envs rollouts)
        agg = {}
        for key in episodes[0]:
            vals = [e[key] for e in episodes if isinstance(e[key], (int, float)) and e[key] is not None]
            if vals:
                agg[key] = float(np.mean(vals))
        agg['n_episodes'] = len(episodes)

        result = {
            'meta': dict(meta, checkpoint=os.path.relpath(ckpt, REPO), epoch=epoch, frame=frame,
                         mode=a.mode, num_envs=num_envs, eval_seed=a.seed, eval_cfg=a.cfg_env,
                         start_frame=a.start_frame,
                         eval_cfg_sha1=cfg_sha, motion_file=a.motion_file, robot_type=a.robot_type,
                         task_eps_m=TASK_EPS_M, git=commit,
                         evaluated_at=datetime.datetime.now().isoformat(timespec='seconds'),
                         eval_seconds=round(time.time() - t0, 1)),
            'aggregate': agg,
            'episodes': episodes,
        }
        path = out_path(meta, ckpt)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(result, f, indent=1)
        if not a.no_series:
            np.savez_compressed(path[:-5] + '.npz', **series)

        mismatch = sum(e['official_success'] != e['official_success_env_flag'] for e in episodes)
        print(f"[eval {k + 1}/{len(jobs)}] {meta['run_name']} epoch={epoch} | "
              f"success={agg['official_success']:.2f} task_success={agg['task_success']:.2f} "
              f"progress={agg['progress']:.3f} obj_pos_err={agg['obj_pos_err_mean']:.4f}m "
              f"(local {agg['obj_pos_err_local_mean']:.4f}m) wrist_pos_win={agg.get('wrist_pos_err_window', float('nan')):.4f}m "
              f"fell={agg['fell']:.2f} | {time.time() - t0:.1f}s -> {os.path.relpath(path, REPO)}")
        if mismatch:
            print(f'[eval][WARN] {mismatch} episode(s): recorder failure flag != env terminate flag')


if __name__ == '__main__':
    main()
