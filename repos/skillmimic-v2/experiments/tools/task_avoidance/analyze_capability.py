"""Donor-prefix handoff diagnostics. PASS refers to controls, never retained whole-task ability."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from core import load_protocol,protocol_hash,assert_initial_equal,unique_state_ids
from analyze import write_csv,metric_names,paired_rows,repeat_check


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    a=parser.parse_args(argv);root=a.input.resolve();out=a.output or root/'analysis'
    if out.exists():raise ValueError('Analysis already exists')
    p=load_protocol(root/'protocol.json');m=json.loads((root/'manifest.json').read_text())
    if m.get('kind')!='capability' or protocol_hash(p)!=m['protocol_hash']:raise ValueError('Wrong/changed capability protocol')
    policies={x['label']:x for x in m['policies']};data={};initials={};allrows=[]
    for job in m['jobs']:
        key=(job['prefix_steps'],job['seed'],job['case'])
        if key in data:raise ValueError('Duplicate capability job')
        folder=root/job['job'];result=json.loads((folder/'result.json').read_text())
        if (result['status']!='COMPLETED' or result['protocol_hash']!=m['protocol_hash'] or
            result['checkpoint_sha256']!=policies[job['policy']]['sha256'] or
            result['prefix_sha256']!=policies['donor']['sha256'] or result['seed']!=job['seed'] or
            result['prefix_steps']!=job['prefix_steps'] or result['diagnostic_reward_match']!='PASS' or
            result['control_no_switch']!=(job['case']=='direct')):raise ValueError('Capability result identity mismatch')
        with np.load(folder/'initial.npz',allow_pickle=False) as z:init=dict(z)
        group=key[:2]
        if group in initials:assert_initial_equal(initials[group],init,p['initial_atol'])
        else:initials[group]=init
        if len(result['episodes'])!=p['num_envs']:raise ValueError('Wrong episode count')
        data[key]=(folder,result['episodes'])
        if not job['repeat']:
            allrows += [dict(prefix_steps=key[0],eval_seed=key[1],case=key[2],**row) for row in result['episodes']]
    controls=[];differences=[];means=[];counts=[]
    metrics=metric_names(p)
    for step in p['capability_prefix_steps']:
        for seed in p['eval_seeds']:
            keys=[(step,seed,case) for case in ['direct','reload','candidate']]
            if any(k not in data for k in keys):raise ValueError('Missing capability case')
            direct,reload,candidate=[data[k] for k in keys]
            control=repeat_check(direct[0],reload[0],p)
            control.update(prefix_steps=step,eval_seed=seed,test='donor_reload_vs_uninterrupted')
            controls.append(control)
            if seed==p['eval_seeds'][0]:
                key=(step,seed,'candidate_repeat')
                if key not in data:raise ValueError('Candidate repeat missing')
                check=repeat_check(candidate[0],data[key][0],p)
                check.update(prefix_steps=step,eval_seed=seed,test='candidate_repeat');controls.append(check)
            valid=[r for r in direct[1] if r['eligible']]
            other=[r for r in candidate[1] if r['eligible']]
            ids=unique_state_ids(initials[step,seed],p['num_envs'])
            counts.append(dict(prefix_steps=step,eval_seed=seed,total=p['num_envs'],eligible=len(valid),unique_states=len({ids[r['env_id']] for r in valid})))
            if control['status']=='PASS' and valid:
                delta=paired_rows(valid,other,metrics)
                differences += [dict(prefix_steps=step,eval_seed=seed,**r) for r in delta]
            for case,rows in [('direct',valid),('candidate',other)]:
                for stage in sorted({r['starting_stage'] for r in rows}):
                    subset=[r for r in rows if r['starting_stage']==stage]
                    lifted=[r['maintained_initial_lift'] for r in subset if r['maintained_initial_lift'] is not None]
                    means.append(dict(prefix_steps=step,eval_seed=seed,case=case,starting_stage=stage,n=len(subset),
                                      n_initially_lifted=len(lifted),maintained_initial_lift_rate=float(np.mean(lifted)) if lifted else None,
                                      **{key:float(np.mean([r[key] for r in subset])) for key in metrics}))
    passed=bool(controls) and all(c['status']=='PASS' for c in controls)
    out.mkdir(parents=True);write_csv(out/'episodes.csv',allrows);write_csv(out/'stage_means.csv',means)
    write_csv(out/'eligibility.csv',counts);write_csv(out/'controls.csv',controls)
    # Any failed control invalidates the paired conclusion bundle. Raw descriptive data remain available.
    if passed:write_csv(out/'paired_differences.csv',differences)
    report=dict(status='PASS' if passed and differences else 'INCONCLUSIVE',execution_status='COMPLETED',
                hypothesis_status='NOT TESTED',controls=controls,eligible_counts=counts,
                interpretation='Conditional tail capability after a donor prefix; does not establish full-task ability or intentional avoidance.',
                thresholds=p.get('threshold_status'),primary_rule='All self-reload and candidate-repeat controls must pass before paired differences are released.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(out);return 0 if report['status']=='PASS' else 3


if __name__=='__main__':sys.exit(main())
