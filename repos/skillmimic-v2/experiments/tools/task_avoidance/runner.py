"""Task Avoidance A/B diagnostics and C0/C1 branching; no GPU imports in orchestration."""
import argparse
import contextlib
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('progress_runner',HERE.parent/'progress_reward/runner.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
from core import load_protocol, protocol_hash
ROOT=P.ROOT
DEFAULT=ROOT/'experiments/configs/task_avoidance/protocol_v2.json'
OUT=ROOT/'experiments/runs/task_avoidance'


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def guards(label):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',label):raise ValueError('Invalid label')
    for key in ['SMV2_CFG_ENV','SMV2_TRAIN_CFG','SMV2_HIST_CKPT']:
        if os.environ.get(key):raise ValueError('Unset '+key+' to preserve frozen fix')


def gpu_env():
    env=os.environ.copy()
    for key in ['SMV2_PROGRESS_WEIGHT','SMV2_PROGRESS_CLIP','SMV2_PROGRESS_LOG','SMV2_RUN_NAME']:
        env.pop(key,None)
    env['PYTHONUNBUFFERED']='1'
    return env


def epoch(path):
    m=re.search(r'_(\d{8})\.pth$',Path(path).name)
    if not m:raise ValueError('Numbered checkpoint required: '+str(path))
    return int(m[1])


def plan_evaluation(a,p):
    if a.run:
        run=Path(a.run).expanduser().resolve()
        meta=json.loads((run/'run_meta.json').read_text())
        if meta.get('clip')!='place_book' or meta.get('method') not in ['fix','fix_progress']:
            raise ValueError('Place Book fix-compatible run required')
        found={epoch(x):x for x in P.numbered(run)}
        missing=set(p['epochs'])-set(found)
        if missing:raise ValueError('Missing requested epochs: '+str(sorted(missing)))
        checkpoints=[found[e] for e in p['epochs']]
    else:checkpoints=[Path(x).expanduser().resolve() for x in a.checkpoints]
    if not checkpoints or any(not x.is_file() for x in checkpoints):raise ValueError('Policy files missing')
    labels=a.policy_labels.split(',') if a.policy_labels else [f'e{epoch(x)}' for x in checkpoints]
    if len(labels)!=len(checkpoints) or len(set(labels))!=len(labels):raise ValueError('Supply unique --policy-labels for same-epoch policies')
    for label in labels:guards(label)
    return [dict(label=label,checkpoint=str(ck),epoch=epoch(ck),sha256=P.sha(ck)) for label,ck in zip(labels,checkpoints)]


def evaluation(a):
    guards(a.label);p=load_protocol(a.protocol);policies=plan_evaluation(a,p)
    if getattr(a,'pair',None) and (len(set(a.pair))!=2 or not set(a.pair)<={x['label'] for x in policies}):
        raise ValueError('--pair must name two different supplied policies')
    out=OUT/('eval_'+a.label)
    commands=[]
    for seed in p['eval_seeds']:
        reference=out/f'{policies[0]["label"]}_s{seed}'/'initial.npz'
        for i,policy in enumerate(policies):
            job=f'{policy["label"]}_s{seed}'
            cmd=[P.python_path(),str(HERE/'evaluate_worker.py'),'--checkpoint',policy['checkpoint'],'--checkpoint-sha',policy['sha256'],
                 '--protocol',str(out/'protocol.json'),'--output',str(out/job),'--seed',str(seed)]
            if i:cmd+=['--reference-initial',str(reference)]
            commands.append(dict(job=job,policy=policy['label'],seed=seed,command=cmd,repeat=False))
        if p['repeat_probe'] and seed==p['eval_seeds'][0]:
            for policy in policies:
                job=f'{policy["label"]}_s{seed}_repeat'
                cmd=[P.python_path(),str(HERE/'evaluate_worker.py'),'--checkpoint',policy['checkpoint'],'--checkpoint-sha',policy['sha256'],
                     '--protocol',str(out/'protocol.json'),'--output',str(out/job),'--seed',str(seed),
                     '--reference-initial',str(reference)]
                commands.append(dict(job=job,policy=policy['label'],seed=seed,command=cmd,repeat=True))
    for job in commands:print(shlex.join(job['command']))
    if a.dry_run:
        print('DRY RUN: protocol copy and workers would be created only during execution');return 0
    pair_args=['--left',a.pair[0],'--right',a.pair[1]] if getattr(a,'pair',None) else []
    return execute_bundle(out,p,policies,commands,'analyze.py',pair_args)


def execute_bundle(out,p,policies,commands,analysis_script,analysis_args=(),kind='evaluation'):
    P.preflight()
    out.parent.mkdir(parents=True,exist_ok=True)
    P.RUNS.mkdir(parents=True,exist_ok=True)
    with P.gpu_lock():
        out.mkdir() # fail if it already exists
        P.write_json(out/'protocol.json',p)
        m=dict(status='RUNNING',started_at=now(),protocol_hash=protocol_hash(p),policies=policies,
               kind=kind,jobs=commands,files=P.manifest_files(),commit=P.capture(['git','rev-parse','HEAD']),
               worktree=P.capture(['git','status','--short']),
               gpu=P.capture(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']),
               packages=P.capture([P.python_path(),'-m','pip','freeze']))
        P.write_json(out/'manifest.json',m)
        code=1
        try:
            for job in commands:
                code=1  # each launch can raise before assigning its return code
                job['status']='RUNNING'
                job['started_at']=now()
                code=P.execute_child(job['command'],out/(job['job']+'.log'),gpu_env())
                job.update(exit_code=code,ended_at=now(),status='COMPLETED' if code==0 else 'FAILED')
                P.write_json(out/'manifest.json',m)
                if code:return code
            code=1
            code=P.execute_child([sys.executable,str(HERE/analysis_script),'--input',str(out)]+list(analysis_args),out/'analysis.log',os.environ.copy())
            return code
        except BaseException as error:
            code=130 if isinstance(error,KeyboardInterrupt) else 1
            m['error']=type(error).__name__+': '+str(error)
            if 'job' in locals() and job.get('status')=='RUNNING':job.update(status='FAILED',error=m['error'])
            raise
        finally:
            m.update(status='COMPLETED' if code==0 else 'INCONCLUSIVE' if code==3 else 'FAILED',exit_code=code,ended_at=now())
            P.write_json(out/'manifest.json',m)


def capability(a):
    guards(a.label);p=load_protocol(a.protocol)
    if p['version']!='avoidance_v2':raise ValueError('Capability probe requires v2 protocol')
    if not p['repeat_probe']:raise ValueError('Capability probe requires candidate repeats')
    source=Path(a.source).expanduser().resolve();candidate=Path(a.candidate).expanduser().resolve()
    for path in [source,candidate]:
        if not path.is_file():raise ValueError('Missing checkpoint: '+str(path))
    policies=[dict(label='donor',checkpoint=str(source),sha256=P.sha(source),epoch=epoch(source)),
              dict(label='candidate',checkpoint=str(candidate),sha256=P.sha(candidate),epoch=epoch(candidate))]
    out=OUT/('capability_'+a.label);commands=[]
    for step in p['capability_prefix_steps']:
        for seed in p['eval_seeds']:
            reference=out/f'prefix{step}_direct_s{seed}'/'initial.npz'
            cases=[('direct',policies[0],True,False),('reload',policies[0],False,False),('candidate',policies[1],False,False)]
            if p['repeat_probe'] and seed==p['eval_seeds'][0]:cases.append(('candidate_repeat',policies[1],False,True))
            for label,policy,no_switch,repeat in cases:
                job=f'prefix{step}_{label}_s{seed}'
                cmd=[P.python_path(),str(HERE/'evaluate_worker.py'),'--checkpoint',policy['checkpoint'],'--checkpoint-sha',policy['sha256'],
                     '--prefix-checkpoint',str(source),'--prefix-sha',policies[0]['sha256'],'--prefix-steps',str(step),
                     '--protocol',str(out/'protocol.json'),'--output',str(out/job),'--seed',str(seed)]
                if label!='direct':cmd+=['--reference-initial',str(reference)]
                if no_switch:cmd+=['--control-no-switch']
                commands.append(dict(job=job,policy=policy['label'],case=label,seed=seed,prefix_steps=step,command=cmd,repeat=repeat))
    for job in commands:print(shlex.join(job['command']))
    if a.dry_run:return 0
    return execute_bundle(out,p,policies,commands,'analyze_capability.py',kind='capability')


def verify(a):
    guards(a.label);p=load_protocol(a.protocol)
    p['num_envs']=4  # explicit verification fixture, not a training/evaluation result
    source=Path(a.checkpoint).expanduser().resolve()
    if not source.is_file():raise ValueError('Checkpoint missing')
    policy=dict(label='checks',checkpoint=str(source),sha256=P.sha(source),epoch=epoch(source))
    out=OUT/('verify_'+a.label)
    cmd=[P.python_path(),str(HERE/'gpu_checks.py'),'--checkpoint',str(source),'--checkpoint-sha',policy['sha256'],
         '--protocol',str(out/'protocol.json'),'--output',str(out/'checks')]
    print(shlex.join(cmd))
    if a.dry_run:return 0
    return execute_bundle(out,p,[policy],[dict(job='checks',command=cmd)],'verify_summary.py',kind='gpu_verification')


def branch(a):
    guards(a.label);p=load_protocol(a.protocol)
    source=Path(a.source).expanduser().resolve()
    if not source.is_file():raise ValueError('Source checkpoint missing')
    source_epoch=epoch(source)
    provenance=json.loads((source.parent.parent/'run_meta.json').read_text())
    if provenance.get('clip')!='place_book' or provenance.get('method')!='fix' or provenance.get('num_envs')!=2048:
        raise ValueError('Branch source must have run_meta: place_book / fix / num_envs=2048')
    extra=a.extra_epochs if a.extra_epochs is not None else p['branch_extra_epochs']
    if extra<=0 or extra%10 or source_epoch%10:raise ValueError('Source epoch and positive extra-epochs must align with 10-epoch saves')
    if a.stage=='smoke' and extra>100:raise ValueError('smoke requires --extra-epochs <=100')
    if not a.dry_run and not a.execute:raise ValueError('Branch training requires --execute or --dry-run')
    seeds=[int(x) for x in a.seeds.split(',')] if a.seeds else p['branch_seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):raise ValueError('Invalid branch seeds')
    source_sha=P.sha(source)
    if not a.dry_run:P.preflight();P.RUNS.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        for variant in ['baseline','progress']:
            weight=0 if variant=='baseline' else p['progress_weight']
            run=OUT/f'branch_{a.label}_{a.stage}_{variant}_s{seed}_from{source_epoch}_plus{extra}'
            args=SimpleNamespace(num_envs=2048,epochs=source_epoch+extra-1,seed=seed)
            run_cmd=P.command(args,run,source)
            run_cmd[run_cmd.index('--output_path')+1]=str(OUT)+'/'
            cmd=[run_cmd[0],str(HERE/'branch_entry.py'),'--source',str(source),'--seed',str(seed),
                 '--source-sha',source_sha,'--source-epoch',str(source_epoch),'--report',str(run/'restore.jsonl'),
                 '--']+run_cmd[2:]
            print('BRANCH',variant,'weight=',weight,'extra_samples=',extra*2048*32)
            print(shlex.join(cmd))
            if a.dry_run:continue
            with P.gpu_lock():
                files=P.manifest_files()
                signature=dict(source_sha256=source_sha,source_epoch=source_epoch,protocol=p,variant=variant,
                               weight=weight,branch_seed=seed,extra_epochs=extra,files=files,python=P.python_path())
                fp=protocol_hash(signature)
                mp=run/'experiment.json'
                if run.exists():
                    if not a.resume or not mp.exists():raise ValueError('Run exists; explicit matching --resume required')
                    m=json.loads(mp.read_text())
                    if m['fingerprint']!=fp:raise ValueError('Branch fingerprint changed')
                    checkpoints=P.numbered(run)
                    if not checkpoints:raise ValueError('No branch checkpoint; restart with a new label')
                    latest=epoch(checkpoints[-1]);target=source_epoch+extra
                    if latest>target:raise ValueError('Checkpoint exceeds target; refusing misleading completion')
                    if latest==target:
                        # Crash after saving the target must NEVER start one more epoch.
                        audit=[P.python_path(),str(HERE/'checkpoints.py'),'--source',str(source),
                               '--source-sha',source_sha,'--checkpoint',str(checkpoints[-1]),
                               '--source-epoch',str(source_epoch),'--target-epoch',str(target),
                               '--output',str(run/'completion.json')]
                        m['status']='VERIFYING';P.write_json(mp,m);code=1
                        try:
                            code=P.execute_child(audit,run/'checkpoint_audit.log',gpu_env())
                            if code==0:
                                completion=json.loads((run/'completion.json').read_text())
                                if completion['status']!='PASS' or completion['final_epoch']!=target:raise ValueError('Invalid completion audit')
                                m['checkpoint_frames']=completion
                        except BaseException as error:
                            code=1;m['error']=type(error).__name__+': '+str(error);raise
                        finally:
                            m.update(status='COMPLETED' if code==0 else 'FAILED',last_saved_epoch=latest,
                                     extra_saved_samples=extra*2048*32,completion_audit_exit_code=code)
                            P.write_json(mp,m)
                        if code:return code
                        continue
                    if m['status']=='COMPLETED':raise ValueError('Completion record contradicts checkpoint epoch')
                    cmd.insert(cmd.index('--'),'--resume')
                    cmd[cmd.index('--resume_from')+1]=str(checkpoints[-1])
                else:
                    if a.resume:raise ValueError('Cannot resume missing branch')
                    run.mkdir(parents=True)
                    m=dict(status='CREATED',fingerprint=fp,spec=signature,segments=[],source_meta=provenance,
                           commit=P.capture(['git','rev-parse','HEAD']),worktree=P.capture(['git','status','--short']),
                           packages=P.capture([P.python_path(),'-m','pip','freeze']),
                           gpu=P.capture(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']),
                           independent_training_seed=False,replication_unit='continuation RNG seed conditional on one source checkpoint')
                    P.write_json(run/'protocol.json',p)
                    P.write_json(run/'run_meta.json',dict(run_name=run.name,clip='place_book',method='fix_progress',
                                                        seed=seed,num_envs=2048,source_checkpoint=str(source)))
                segment=dict(started_at=now(),command=cmd);m['segments'].append(segment);m['status']='RUNNING'
                P.write_json(mp,m)
                env=gpu_env();env.update(SMV2_RUN_NAME=run.name,SMV2_PROGRESS_WEIGHT=str(weight),
                    SMV2_PROGRESS_CLIP=str(p['progress_clip']),SMV2_PROGRESS_LOG=str(run/'reward_components.jsonl'))
                code=1
                try:
                    code=P.execute_child(cmd,run/'train.log',env)
                    if code==0:
                        completion=json.loads((run/'completion.json').read_text())
                        if completion.get('status')!='PASS' or completion.get('final_epoch')!=source_epoch+extra:
                            raise ValueError('Training ended without verified completion')
                except BaseException as error:
                    code=130 if isinstance(error,KeyboardInterrupt) else 1
                    m['error']=type(error).__name__+': '+str(error)
                    raise
                finally:
                    ckpts=P.numbered(run);last=epoch(ckpts[-1]) if ckpts else source_epoch
                    if (run/'completion.json').exists():m['checkpoint_frames']=json.loads((run/'completion.json').read_text())
                    m.update(status='COMPLETED' if code==0 and last>=source_epoch+extra else 'FAILED',
                             last_saved_epoch=last,extra_saved_samples=max(0,last-source_epoch)*2048*32,
                             evaluation_status='NOT RUN')
                    segment.update(ended_at=now(),exit_code=code);P.write_json(mp,m)
                if code:return code
                if m['status']!='COMPLETED':return 1
    return 0


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    e=sub.add_parser('evaluate');g=e.add_mutually_exclusive_group(required=True)
    g.add_argument('--run');g.add_argument('--checkpoints',nargs='+')
    e.add_argument('--policy-labels',default='')
    e.add_argument('--pair',nargs=2,metavar=('LEFT','RIGHT'))
    v=sub.add_parser('verify');v.add_argument('--checkpoint',required=True)
    c=sub.add_parser('capability');c.add_argument('--source',required=True);c.add_argument('--candidate',required=True)
    b=sub.add_parser('branch');b.add_argument('--source',required=True)
    b.add_argument('--stage',choices=['smoke','main'],default='smoke');b.add_argument('--extra-epochs',type=int)
    b.add_argument('--seeds',default='');b.add_argument('--execute',action='store_true');b.add_argument('--resume',action='store_true')
    for q in [e,b,c,v]:
        q.add_argument('--protocol',type=Path,default=DEFAULT)
        q.add_argument('--label',required=True);q.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(argv)
    return {'evaluate':evaluation,'branch':branch,'capability':capability,'verify':verify}[a.action](a)


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        print('ERROR:',e,file=sys.stderr);sys.exit(2)
