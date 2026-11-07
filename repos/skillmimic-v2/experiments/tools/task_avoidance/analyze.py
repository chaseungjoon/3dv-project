"""CPU-only paired evaluation analysis; uncertainty is NOT independent-training uncertainty."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import numpy as np
from core import load_protocol, protocol_hash, assert_initial_equal, transition_masks

METRICS=['return_train_window','return_eval','lift_attempt','lift_success','goal_progress_m',
         'fall','stable_lift','task_success','observed_steps','wrist_near_steps','held_steps']


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def paired_rows(left,right):
    x={r['env_id']:r for r in left};y={r['env_id']:r for r in right}
    if x.keys()!=y.keys():raise ValueError('Paired env IDs differ')
    return [dict(env_id=e,**{m:y[e][m]-x[e][m] for m in METRICS}) for e in sorted(x)]


def repeat_check(left,right,p):
    with np.load(left/'trajectory.npz',allow_pickle=False) as x,np.load(right/'trajectory.npz',allow_pickle=False) as y:
        if set(x)!=set(y) or any(x[k].shape!=y[k].shape for k in x):
            raise ValueError('Repeat trajectory shapes differ')
        if any(not np.isfinite(x[k]).all() or not np.isfinite(y[k]).all() for k in x):
            raise ValueError('Nonfinite repeat trajectory')
        # Compare only the originally evaluated episode, including its terminal transition.
        vx,_=transition_masks(x['done'],x['train_stop']);vy,_=transition_masks(y['done'],y['train_stop'])
        masks_equal=np.array_equal(vx,vy)
        valid=vx|vy
        pos=float(np.max(np.abs(x['obj'][valid]-y['obj'][valid])))
        reward=float(np.max(np.abs(x['reward'][valid]-y['reward'][valid])))
        flags=all(np.array_equal(x[k][valid],y[k][valid]) for k in ['done','fall','train_stop'])
        passed=masks_equal and flags and pos<=p['repeat_position_atol_m'] and reward<=p['repeat_reward_atol']
        return dict(status='PASS' if passed else 'INCONCLUSIVE',max_position_abs_m=pos,
                    max_reward_abs=reward,termination_equal=flags,valid_masks_equal=masks_equal)


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
    results={};initials={};rows=[];checks=[]
    for job in m['jobs']:
        folder=root/job['job'];r=json.loads((folder/'result.json').read_text())
        if r['status']!='COMPLETED' or r['protocol_hash']!=ph or r['diagnostic_reward_match']!='PASS':
            raise ValueError('Incomplete/invalid result: '+job['job'])
        with np.load(folder/'initial.npz',allow_pickle=False) as z:init=dict(z)
        seed=job['seed'];label=job['policy']
        if seed in initials:assert_initial_equal(initials[seed],init,p['initial_atol'])
        else:initials[seed]=init
        if job['repeat']:
            original=root/f'{label}_s{seed}'
            checks.append(repeat_check(original,folder,p));continue
        if len(r['episodes'])!=p['num_envs']:raise ValueError('Incomplete episode count')
        if (label,seed) in results:raise ValueError('Duplicate policy/seed result')
        results[label,seed]=r['episodes']
        for episode in r['episodes']:rows.append(dict(policy=label,eval_seed=seed,**episode))
    for policy in m['policies']:
        for seed in p['eval_seeds']:
            if (policy['label'],seed) not in results:raise ValueError('Missing policy/seed evaluation')
    if p['repeat_probe'] and not checks:raise ValueError('Repeat probe missing')
    reliable=bool(checks) and all(x['status']=='PASS' for x in checks)
    out.mkdir(parents=True)
    write_csv(out/'episodes.csv',rows)
    means=[]
    for (label,seed),episodes in results.items():
        means.append(dict(policy=label,eval_seed=seed,**{key:float(np.mean([r[key] for r in episodes])) for key in METRICS}))
    write_csv(out/'per_eval_seed.csv',means)
    labels=[pol['label'] for pol in m['policies']]
    left=a.left or f'e{p["primary_pair"][0]}';right=a.right or f'e{p["primary_pair"][1]}'
    paired=[];per_seed=[]
    if reliable and left in labels and right in labels:
        for seed in p['eval_seeds']:
            delta=paired_rows(results[left,seed],results[right,seed])
            paired += [dict(left=left,right=right,eval_seed=seed,**r) for r in delta]
            per_seed.append(dict(eval_seed=seed,**{key:float(np.mean([r[key] for r in delta])) for key in METRICS}))
        write_csv(out/'paired_episode_differences.csv',paired)
        write_csv(out/'paired_seed_means.csv',per_seed)
    # Per-policy descriptive chart; points averaged over evaluation seeds, NOT learning replicates.
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640">',
           '<rect width="100%" height="100%" fill="white"/>']
    for j,metric in enumerate(['return_train_window','lift_attempt','goal_progress_m','fall']):
        y=30+j*150;values=[float(np.mean([r[metric] for r in means if r['policy']==label])) for label in labels]
        lo=min(0,min(values));hi=max(values);span=max(hi-lo,1e-8)
        parts.append(f'<text x="20" y="{y}">{metric}; range {lo:.3g} to {hi:.3g}</text>')
        points=[]
        for i,(label,value) in enumerate(zip(labels,values)):
            x=100+i*760/max(len(labels)-1,1);py=y+100-80*(value-lo)/span;points.append(f'{x},{py}')
            parts.append(f'<circle cx="{x}" cy="{py}" r="4" fill="#2563eb"/><text x="{x-25}" y="{y+120}" font-size="11">{html.escape(label)}</text>')
        parts.append('<polyline points="'+' '.join(points)+'" fill="none" stroke="#2563eb"/>')
    parts.append('</svg>');(out/'curves.svg').write_text('\n'.join(parts))
    report=dict(status='PASS' if reliable else 'INCONCLUSIVE',
                meaning='PASS means pairing/repeat diagnostics passed, NOT that avoidance is proven',
                initial_pairing='PASS',repeat_checks=checks,primary_pair=[left,right],
                paired_comparison='AVAILABLE' if paired else 'NOT RUN',
                differences_direction='right minus left',per_eval_seed_differences=per_seed,
                limitations=['One source policy training seed is not independent learning replication.',
                             'Training-rule-censored finite return is not the bootstrapped PPO objective.',
                             'No automatic significance or Task Avoidance verdict.',
                             'Repeat probe covers only the first checkpoint and first evaluation seed.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out);return 0 if reliable else 3


if __name__=='__main__':sys.exit(main())
