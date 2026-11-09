"""CPU-only paired evaluation analysis; uncertainty is NOT independent-training uncertainty."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import numpy as np
from core import load_protocol, protocol_hash, assert_initial_equal, transition_masks, unique_state_ids

METRICS=['return_train_window','return_eval','lift_attempt','lift_success','goal_progress_m',
         'fall','stable_lift','task_success','observed_steps','wrist_near_steps','held_steps']


METRICS_V2=[m for m in METRICS if m!='lift_attempt']+['partial_lift','reach_attempt_proxy','reach_without_partial_lift']


def metric_names(p):return METRICS_V2 if p['version']=='avoidance_v2' else METRICS


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)


def paired_rows(left,right,metrics=None):
    metrics=metrics or METRICS
    x={r['env_id']:r for r in left};y={r['env_id']:r for r in right}
    if len(x)!=len(left) or len(y)!=len(right):raise ValueError('Duplicate env IDs')
    if x.keys()!=y.keys():raise ValueError('Paired env IDs differ')
    return [dict(env_id=e,**{m:y[e][m]-x[e][m] for m in metrics}) for e in sorted(x)]


def repeat_check(left,right,p):
    with np.load(left/'trajectory.npz',allow_pickle=False) as x,np.load(right/'trajectory.npz',allow_pickle=False) as y:
        if set(x)!=set(y) or any(x[k].shape!=y[k].shape for k in x):
            raise ValueError('Repeat trajectory shapes differ')
        if any(not np.isfinite(x[k]).all() or not np.isfinite(y[k]).all() for k in x):
            raise ValueError('Nonfinite repeat trajectory')
        # Compare only the originally evaluated episode, including its terminal transition.
        vx,_=transition_masks(x['done'],x['train_stop']);vy,_=transition_masks(y['done'],y['train_stop'])
        eligible=np.ones(vx.shape[1],bool)
        valid=vx|vy
        if (left/'initial.npz').exists():
            with np.load(left/'initial.npz',allow_pickle=False) as init:
                if 'eligible' in init:eligible=init['eligible'].astype(bool)
        valid &= eligible[None]
        masks_equal=np.array_equal(vx[:,eligible],vy[:,eligible])
        if not valid.any():return dict(status='INCONCLUSIVE',reason='No eligible transitions')
        pos=float(np.max(np.abs(x['obj'][valid]-y['obj'][valid])))
        wrist=float(np.max(np.abs(x['wrist_pos'][valid]-y['wrist_pos'][valid]))) if p['version']=='avoidance_v2' else 0.0
        reward=float(np.max(np.abs(x['reward'][valid]-y['reward'][valid])))
        flags=all(np.array_equal(x[k][valid],y[k][valid]) for k in ['done','fall','train_stop'])
        passed=masks_equal and flags and max(pos,wrist)<=p['repeat_position_atol_m'] and reward<=p['repeat_reward_atol']
        return dict(status='PASS' if passed else 'INCONCLUSIVE',max_position_abs_m=pos,
                    max_wrist_position_abs_m=wrist,max_reward_abs=reward,termination_equal=flags,valid_masks_equal=masks_equal)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--left');ap.add_argument('--right')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args(argv);root=a.input.resolve();p=load_protocol(root/'protocol.json')
    m=json.loads((root/'manifest.json').read_text());ph=protocol_hash(p)
    if ph!=m['protocol_hash']:raise ValueError('Protocol changed after execution')
    out=a.output or root/'analysis'
    if out.exists():raise ValueError('Analysis output exists; choose a new --output')
    metrics=metric_names(p)
    results={};initials={};rows=[];checks=[];duplicates=[]
    policies={r['label']:r for r in m['policies']}
    for job in m['jobs']:
        folder=root/job['job'];r=json.loads((folder/'result.json').read_text())
        if r['status']!='COMPLETED' or r['protocol_hash']!=ph or r['diagnostic_reward_match']!='PASS':
            raise ValueError('Incomplete/invalid result: '+job['job'])
        if p['version']=='avoidance_v2':
            if r.get('checkpoint_sha256')!=policies[job['policy']]['sha256'] or r.get('seed')!=job['seed']:
                raise ValueError('Result identity mismatch: '+job['job'])
            if r.get('prefix_steps')!=0:raise ValueError('Use capability analyzer for switched policies')
        with np.load(folder/'initial.npz',allow_pickle=False) as z:init=dict(z)
        seed=job['seed'];label=job['policy']
        if seed in initials:assert_initial_equal(initials[seed],init,p['initial_atol'])
        else:
            initials[seed]=init
            ids=unique_state_ids(init,p['num_envs'])
            duplicates.append(dict(eval_seed=seed,n=len(ids),unique_states=len(set(ids)) if None not in ids else None))
        if job['repeat']:
            original=root/f'{label}_s{seed}'
            checks.append(dict(policy=label,**repeat_check(original,folder,p)));continue
        if len(r['episodes'])!=p['num_envs']:raise ValueError('Incomplete episode count')
        if (label,seed) in results:raise ValueError('Duplicate policy/seed result')
        results[label,seed]=r['episodes']
        for episode in r['episodes']:rows.append(dict(policy=label,eval_seed=seed,**episode))
    for policy in m['policies']:
        for seed in p['eval_seeds']:
            if (policy['label'],seed) not in results:raise ValueError('Missing policy/seed evaluation')
    if p['repeat_probe'] and not checks:raise ValueError('Repeat probe missing')
    required=set(policies) if p['version']=='avoidance_v2' else {m['policies'][0]['label']}
    reliable=required<={c['policy'] for c in checks} and all(x['status']=='PASS' for x in checks)
    out.mkdir(parents=True)
    write_csv(out/'episodes.csv',rows)
    write_csv(out/'initial_state_counts.csv',duplicates)
    means=[]
    for (label,seed),episodes in results.items():
        means.append(dict(policy=label,eval_seed=seed,**{key:float(np.mean([r[key] for r in episodes])) for key in metrics}))
    write_csv(out/'per_eval_seed.csv',means)
    labels=[pol['label'] for pol in m['policies']]
    left=a.left or f'e{p["primary_pair"][0]}';right=a.right or f'e{p["primary_pair"][1]}'
    paired=[];per_seed=[]
    if reliable and left in labels and right in labels:
        for seed in p['eval_seeds']:
            delta=paired_rows(results[left,seed],results[right,seed],metrics)
            paired += [dict(left=left,right=right,eval_seed=seed,**r) for r in delta]
            per_seed.append(dict(eval_seed=seed,**{key:float(np.mean([r[key] for r in delta])) for key in metrics}))
        write_csv(out/'paired_episode_differences.csv',paired)
        write_csv(out/'paired_seed_means.csv',per_seed)
    # Per-policy descriptive chart; points averaged over evaluation seeds, NOT learning replicates.
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640">',
           '<rect width="100%" height="100%" fill="white"/>']
    for j,metric in enumerate(['return_train_window','reach_attempt_proxy' if p['version']=='avoidance_v2' else 'lift_attempt','goal_progress_m','fall']):
        y=30+j*150;values=[float(np.mean([r[metric] for r in means if r['policy']==label])) for label in labels]
        lo=min(0,min(values));hi=max(values);span=max(hi-lo,1e-8)
        parts.append(f'<text x="20" y="{y}">{metric}; range {lo:.3g} to {hi:.3g}</text>')
        points=[]
        for i,(label,value) in enumerate(zip(labels,values)):
            x=100+i*760/max(len(labels)-1,1);py=y+100-80*(value-lo)/span;points.append(f'{x},{py}')
            parts.append(f'<circle cx="{x}" cy="{py}" r="4" fill="#2563eb"/><text x="{x-25}" y="{y+120}" font-size="11">{html.escape(label)}</text>')
        parts.append('<polyline points="'+' '.join(points)+'" fill="none" stroke="#2563eb"/>')
    parts.append('</svg>');(out/'curves.svg').write_text('\n'.join(parts))
    report=dict(status='PASS' if reliable and paired else 'INCONCLUSIVE',
                execution_status='COMPLETED',hypothesis_status='NOT TESTED',initial_state_counts=duplicates,
                meaning='PASS means pairing/repeat diagnostics passed, NOT that avoidance is proven',
                initial_pairing='PASS',repeat_checks=checks,primary_pair=[left,right],
                paired_comparison='AVAILABLE' if paired else 'NOT RUN',
                differences_direction='right minus left',per_eval_seed_differences=per_seed,
                limitations=['One source policy training seed is not independent learning replication.',
                             'Training-rule-censored finite return is not the bootstrapped PPO objective.',
                             'No automatic significance or Task Avoidance verdict.',
                             'Repeat probe covers each checkpoint at only the first evaluation seed (v2).' ])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(out);return 0 if reliable and paired else 3


if __name__=='__main__':sys.exit(main())
