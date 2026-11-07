"""CPU-friendly orchestration; never imports torch/Isaac Gym in dry-run mode."""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
GITROOT = ROOT.parents[1]
RUNS = ROOT / 'experiments/runs/progress_reward'


def fail(message):
    raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def capture(args):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else 'UNAVAILABLE'


def python_path():
    p = os.environ.get('PY', str(ROOT / '.venv/bin/python'))
    found = shutil.which(p)
    return str(Path(found or p).expanduser().resolve())


def required():
    return [ROOT/'experiments/checkpoints/hist_encoder/parahome_hist60.ckpt',
            ROOT/'skillmimic/data/motions/ParaHome/place_book/002_s6_place_book_2460_2610.pt',
            ROOT/'skillmimic/data/motions/ParaHome/s10/object_transformations.pkl',
            ROOT/'skillmimic/data/assets/obj/book_fix.urdf',
            ROOT/'skillmimic/data/assets/mjcf/mocap_parahome_boxhand_histphase.xml']


def preflight():
    errors = []
    if sys.platform != 'linux':
        errors.append('GPU execution requires the prepared Linux Isaac Gym machine')
    if not Path(python_path()).is_file():
        errors.append('Python missing: set PY to the existing Isaac Gym environment')
    if not shutil.which('nvidia-smi'):
        errors.append('nvidia-smi missing')
    for p in required():
        if not p.is_file():
            errors.append('Missing: ' + str(p.relative_to(ROOT)))
    parent = RUNS
    while not parent.exists():
        parent = parent.parent
    if not os.access(parent, os.W_OK):
        errors.append('Output ancestor is not writable: ' + str(parent))
    free = shutil.disk_usage(parent).free / 2**30
    print(f'Disk free: {free:.1f} GiB; reserve at least 40 GiB for one full A/B pair')
    if errors:
        fail('\n'.join(errors) + '\nNo installation attempted. Restore files from the existing fix machine.')
    print(capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv']))
    subprocess.run([python_path(), '-c', 'import isaacgym; import torch; '
                    'print(torch.__version__, torch.version.cuda); '
                    'assert torch.cuda.is_available(), "CUDA unavailable"'], check=True, cwd=ROOT)
    print('Preflight PASS (not a training or physics validation)')


def manifest_files():
    files = []
    for folder in ['skillmimic', 'experiments/configs', 'experiments/scripts', 'experiments/tools']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix not in ['.pyc']]
    files += required()
    return {str(p.relative_to(ROOT)): sha(p) if p.is_file() else 'MISSING'
            for p in sorted(set(files))}


def validate(a):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.label):
        fail('label must use letters, digits, underscores or hyphens')
    if a.epochs <= 0 or a.seed < 0 or a.num_envs <= 0:
        fail('epochs/num-envs must be positive and seed nonnegative')
    if (a.num_envs * 32) % 16384:
        fail('num-envs * 32 must be divisible by the fixed minibatch size 16384')
    if not math.isfinite(a.weight) or a.weight < 0:
        fail('weight must be finite and nonnegative')
    if a.stage == 'smoke' and a.epochs > 100:
        fail('smoke is limited to 100 epochs; use --stage main explicitly')
    if a.stage == 'main' and not a.execute and not a.dry_run:
        fail('Main training requires --execute (or use --dry-run)')
    for k in ['SMV2_CFG_ENV', 'SMV2_TRAIN_CFG', 'SMV2_HIST_CKPT']:
        if os.environ.get(k):
            fail(k + ' override would change the frozen configuration; unset it')


def identity(a):
    w = 0.0 if a.variant == 'baseline' else a.weight
    return f'book_{a.label}_{a.stage}_{a.variant}_w{w:g}_s{a.seed}_n{a.num_envs}_e{a.epochs}_scratch'


def command(a, run, resume=None):
    args = [python_path(), 'skillmimic/run.py', '--task', 'SkillMimicParahomeProgress',
            '--cfg_env', 'experiments/configs/env/place_book_fix.yaml',
            '--cfg_train', 'experiments/configs/train/parahome.yaml',
            '--asset_file_name', 'mjcf/mocap_parahome_boxhand_histphase.xml',
            '--motion_file', 'skillmimic/data/motions/ParaHome/place_book',
            '--hist_length', '60', '--history_embedding_size', '3',
            '--hist_ckpt', str(required()[0]), '--episode_length', '60',
            '--reweight', '--reweight_alpha', '1.0', '--state_init_random_prob', '0.1',
            '--enable_buffernode', '--num_envs', str(a.num_envs), '--max_epochs', str(a.epochs),
            '--seed', str(a.seed), '--experiment', run.name, '--output_path', str(RUNS) + '/', '--headless']
    if resume:
        args += ['--resume_from', str(resume)]
    return args


def numbered(run):
    return sorted(p for p in (run/'nn').glob('*.pth') if re.search(r'_\d{8}\.pth$', p.name))


def execute_child(args, log, env):
    """Stream output; forward signals only to our child process group."""
    child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
    received = []
    def forward(sig, frame):
        received.append(sig)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, sig)
    old = {s: signal.signal(s, forward) for s in [signal.SIGINT, signal.SIGTERM]}
    try:
        with open(log, 'a') as f:
            for line in child.stdout:
                f.write(line)
                f.flush()
                print(line, end='', flush=True)
        code = child.wait()
        if received:
            return 128 + received[0]
        return code if code >= 0 else 128 - code
    finally:
        if child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        child.stdout.close()
        for s, handler in old.items():
            signal.signal(s, handler)


@contextlib.contextmanager
def gpu_lock():
    # Linux advisory lock; releases automatically on process exit, including a crash.
    import fcntl
    lock = RUNS / '.gpu.lock'
    with open(lock, 'a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail('Another progress experiment/evaluation is running in this checkout')
        yield


def train(a):
    validate(a)
    run = RUNS/identity(a)
    cmd = command(a, run)
    print('RUN:', run)
    print('WEIGHT:', 0 if a.variant == 'baseline' else a.weight)
    print(shlex.join(cmd))
    if a.dry_run:
        print('DRY RUN: no files created; data/GPU not checked; resume checkpoint resolved only at execution')
        return 0
    preflight()
    RUNS.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        config = json.loads((ROOT/'experiments/configs/progress_reward'/f'{a.variant}.json').read_text())
        if config['task'] != 'place_book' or config['goal'] != 'last_reference_object_position':
            fail('Unsupported reward configuration')
        clip = float(config['clip'])
        if not math.isfinite(clip) or clip <= 0:
            fail('Invalid reward clip')
        files = manifest_files()
        spec = dict(task='place_book', variant=a.variant,
                    weight=0 if a.variant == 'baseline' else a.weight, clip=clip,
                    epochs=a.epochs, seed=a.seed, num_envs=a.num_envs,
                    files=files, python=python_path())
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        mp = run/'experiment.json'
        resume = None
        if run.exists():
            if not a.resume:
                fail('Run already exists; use --resume explicitly or choose a new --label')
            if not mp.exists():
                fail('Existing directory lacks experiment.json; refusing to adopt it')
            old = json.loads(mp.read_text())
            if old['fingerprint'] != fingerprint:
                fail('Configuration/source/data fingerprint changed; resume refused')
            if old['training_status'] == 'COMPLETED':
                print('Already COMPLETED; explicit resume is a no-op')
                return 0
            ckpts = numbered(run)
            if not ckpts:
                fail('No numbered checkpoint to resume; use a new label for a fresh run')
            resume = ckpts[-1]
            meta = old
        else:
            if a.resume:
                fail('Cannot resume a missing run')
            run.mkdir()
            meta = dict(id=run.name, fingerprint=fingerprint, spec=spec, segments=[],
                        evaluation_status='NOT RUN',
                        commit=capture(['git', 'rev-parse', 'HEAD']),
                        worktree=capture(['git', 'status', '--short']),
                        gpu=capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv']),
                        python_version=capture([python_path(), '--version']),
                        packages=capture([python_path(), '-m', 'pip', 'freeze']))
        cmd = command(a, run, resume)
        segment = dict(started_at=dt.datetime.now(dt.timezone.utc).isoformat(), command=cmd,
                       resume_from=str(resume) if resume else None)
        meta['segments'].append(segment)
        meta['training_status'] = 'RUNNING'
        meta['target_samples'] = a.epochs*a.num_envs*32
        write_json(mp, meta)
        # Required by the existing evaluator; method maps to the identical observation architecture.
        write_json(run/'run_meta.json', dict(run_name=run.name, clip='place_book', method='fix_progress',
                                           seed=a.seed, num_envs=a.num_envs))
        env = os.environ.copy()
        env.update(SMV2_RUN_NAME=run.name, SMV2_PROGRESS_WEIGHT=str(spec['weight']),
                   SMV2_PROGRESS_CLIP=str(spec['clip']), SMV2_PROGRESS_LOG=str(run/'reward_components.jsonl'),
                   PYTHONUNBUFFERED='1')
        gpu_log = open(run/'gpu.csv', 'a')
        monitor = None
        code = 1
        try:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu',
                                        '--format=csv,noheader', '-l', '10'], stdout=gpu_log,
                                       stderr=subprocess.DEVNULL)
            code = execute_child(cmd, run/'train.log', env)
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait()
            gpu_log.close()
            ckpts = numbered(run)
            epoch = int(ckpts[-1].stem[-8:]) if ckpts else 0
            meta['last_saved_epoch'] = epoch
            meta['last_saved_samples'] = epoch*a.num_envs*32
            meta['training_status'] = ('INTERRUPTED' if code in [130,143] else
                                      'COMPLETED' if code == 0 and epoch >= a.epochs else 'FAILED')
            segment.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), exit_code=code)
            write_json(mp, meta)
        return code if code else (0 if meta['training_status'] == 'COMPLETED' else 1)


def evaluate(a):
    for key in ['SMV2_CFG_ENV', 'SMV2_TRAIN_CFG', 'SMV2_HIST_CKPT']:
        if os.environ.get(key):
            fail(key + ' override must be unset for frozen evaluation')
    run = Path(a.run).expanduser().resolve()
    if not (run/'run_meta.json').is_file():
        fail('run_meta.json missing: '+str(run))
    if not numbered(run):
        fail('Numbered policy checkpoints missing: '+str(run/'nn'))
    if a.epochs:
        wanted = {int(x) for x in a.epochs.split(',')}
        present = {int(p.stem[-8:]) for p in numbered(run)}
        if wanted - present:
            fail('Requested checkpoints missing: ' + str(sorted(wanted - present)))
    seeds = [int(s) for s in a.seeds.split(',')]
    if any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        fail('Evaluation seeds must be distinct and nonnegative')
    commands = []
    for seed in seeds:
        args = [python_path(), 'experiments/tools/evaluate.py', '--run_dir', str(run),
                '--mode', a.mode, '--num_envs', str(a.num_envs), '--seed', str(seed),
                '--out_dir', str(run/'evaluation'/a.label),
                '--tag', f'{a.label}_s{seed}_n{a.num_envs}']
        args += ['--epochs', a.epochs] if a.epochs else ['--which', 'final']
        if a.perturb:
            args += ['--perturb']
        commands.append(args)
        print(shlex.join(args))
    if a.dry_run:
        return 0
    preflight()
    RUNS.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        env = os.environ.copy()
        # Evaluation actions are identical for weight0/weight>0; freeze original evaluator protocol.
        env['SMV2_PROGRESS_WEIGHT'] = '0'
        env.pop('SMV2_PROGRESS_LOG', None)
        record = run/('evaluation_' + a.label + '.json')
        if record.exists():
            fail('Evaluation label exists; choose a new --label to preserve previous results')
        meta = dict(status='RUNNING', protocol='legacy_fix_v1', commands=commands,
                    started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    files=manifest_files(), commit=capture(['git','rev-parse','HEAD']),
                    worktree=capture(['git','status','--short']),
                    note='det refers to actions; original internal initialization noise is preserved')
        write_json(record, meta)
        for args in commands:
            code = execute_child(args, run/('evaluation_' + a.label + '.log'), env)
            if code:
                meta.update(status='FAILED', exit_code=code)
                write_json(record, meta)
                return code
        meta.update(status='COMPLETED', exit_code=0, ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(record, meta)
        p = run/'experiment.json'
        if p.exists():
            m = json.loads(p.read_text()); m['evaluation_status'] = 'COMPLETED'; write_json(p,m)
    return 0


def collect(a):
    run = Path(a.run).resolve()
    if not run.is_dir():
        fail('Run directory missing')
    out = Path(a.output).resolve()
    if out == run or run in out.parents:
        fail('Archive must be outside the run directory')
    # Exclusive creation prevents accidental overwrite.
    with open(out, 'xb') as handle, tarfile.open(fileobj=handle, mode='w:gz') as tar:
        for p in sorted(run.rglob('*')):
            if p.is_symlink():
                continue
            if p.is_file() and (a.checkpoints or p.suffix not in ['.pth','.ckpt']):
                tar.add(p, arcname=str(Path(run.name)/p.relative_to(run)), recursive=False)
    print(out)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('preflight')
    for name in ['train','compare']:
        q = sub.add_parser(name)
        q.add_argument('--variant', choices=['baseline','progress'], default='baseline')
        q.add_argument('--stage', choices=['smoke','main'], default='smoke')
        q.add_argument('--label', default='pilot')
        q.add_argument('--epochs', type=int, default=20)
        q.add_argument('--seed', type=int, default=0)
        q.add_argument('--seeds', default='0', help='compare only')
        q.add_argument('--num-envs', type=int, default=2048)
        q.add_argument('--weight', type=float, default=1.0)
        q.add_argument('--dry-run', action='store_true')
        q.add_argument('--execute', action='store_true')
        q.add_argument('--resume', action='store_true')
    q=sub.add_parser('evaluate')
    q.add_argument('--run', required=True)
    q.add_argument('--seeds', default='101,102,103')
    q.add_argument('--mode', choices=['det','stoch'], default='det')
    q.add_argument('--num-envs', type=int, default=32)
    q.add_argument('--epochs', default='')
    q.add_argument('--label', default='final')
    q.add_argument('--perturb', action='store_true')
    q.add_argument('--dry-run', action='store_true')
    q=sub.add_parser('collect')
    q.add_argument('--run', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--checkpoints', action='store_true')
    a=p.parse_args(argv)
    if hasattr(a,'label') and not re.fullmatch(r'[A-Za-z0-9_-]+',a.label):
        fail('Invalid label')
    if a.action=='preflight':
        preflight(); return 0
    if a.action=='train': return train(a)
    if a.action=='evaluate':
        if a.num_envs<=0: fail('num-envs must be positive')
        return evaluate(a)
    if a.action=='collect': return collect(a)
    seeds=[int(s) for s in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds): fail('Duplicate seeds')
    for seed in seeds:
        for variant in ['baseline','progress']:
            a.seed=seed; a.variant=variant
            code=train(a)
            if code: return code
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print('ERROR:',e,file=sys.stderr)
        sys.exit(2)
