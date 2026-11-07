"""CPU-only JSON/NPZ aggregation. No checkpoints required. Writes a new output directory."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a=p.parse_args()
    if not a.input.is_dir(): raise SystemExit('Input directory missing')
    rows=[]
    for f in sorted(a.input.rglob('*.json')):
        d=json.loads(f.read_text())
        if not all(k in d for k in ['meta','aggregate','episodes']): continue
        m,g=d['meta'],d['aggregate']
        row=dict(run=m.get('run_name',f.parent.name),epoch=m.get('epoch'),
                 mode=m.get('mode'),seed=m.get('seed'),perturb=m.get('perturb'),
                 n=g.get('n_episodes'),lift=g.get('lift_success'),task=g.get('task_success'),
                 fall=g.get('fell_in_clip'),held_frames=g.get('held_frames'),
                 source=str(f),wrist_near_alive_fraction=None,lift_clip=None,stable_lift_clip=None,
                 end_goal_error_survivors_m=None,survivors=None,
                 displacement_toward_goal_m=None,first_tracking_failure_frame=None)
        if f.with_suffix('.npz').exists():
            with np.load(f.with_suffix('.npz'),allow_pickle=False) as z:
                nc=min(len(z['obj']),m['clip_frames']-m['start_frame'])
                if nc<=0: raise ValueError('Invalid clip horizon')
                obj=z['obj'][:nc];ref=z['ref_obj'][:nc];alive=z['alive'][:nc].astype(bool)
                # Post-step height baseline differs from legacy ref start; label this auxiliary metric.
                lifted=((obj[:,:,2]-ref[0,:,2])>=.5*m['ref_lift']) & (z['wrist_d'][:nc]<.2) & alive
                survivors=alive[-1]
                goal=ref[-1]
                ends=[];progress=[];failures=[]
                err=np.linalg.norm(obj-ref,axis=-1)
                for e in range(obj.shape[1]):
                    valid=np.flatnonzero(alive[:,e])
                    if len(valid):
                        end=obj[valid[-1],e]
                        progress.append(float(np.linalg.norm(obj[0,e]-goal[e])-np.linalg.norm(end-goal[e])))
                    bad=np.flatnonzero((err[:,e]>.1)|~alive[:,e])
                    failures.append(int(bad[0]) if len(bad) else nc)
                row.update(wrist_near_alive_fraction=float(((z['wrist_d'][:nc]<.2)&alive).sum()/max(alive.sum(),1)),
                           lift_clip=float(lifted.any(0).mean()),
                           stable_lift_clip=float((lifted.any(0)&alive.all(0)).mean()),
                           survivors=int(survivors.sum()),
                           end_goal_error_survivors_m=float(np.linalg.norm(obj[-1,survivors]-goal[survivors],axis=-1).mean()) if survivors.any() else None,
                           displacement_toward_goal_m=float(np.mean(progress)) if progress else None,
                           first_tracking_failure_frame=float(np.median(failures)))
        rows.append(row)
    if not rows: raise SystemExit('No evaluation JSON files found')
    a.output.mkdir(parents=True,exist_ok=False)
    with open(a.output/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    # Separate panels by run/mode/seed/perturb: never pool independent training seeds silently.
    groups={}
    for r in rows:
        key=(r['run'],r['mode'],r['seed'],r['perturb'])
        groups.setdefault(key,[]).append(r)
    height=230*len(groups)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
           '<rect width="100%" height="100%" fill="white"/>']
    for i,(key,rs) in enumerate(groups.items()):
        y=i*230;rs.sort(key=lambda x:x['epoch']);lo=min(r['epoch'] for r in rs);hi=max(r['epoch'] for r in rs)
        parts.append(f'<text x="40" y="{y+20}" font-size="12">{html.escape(str(key))}</text>')
        parts.append(f'<path d="M 50 {y+45} V {y+190} H 850" fill="none" stroke="black"/>')
        for metric,color in [('lift','#2563eb'),('fall','#dc2626'),('task','#16a34a')]:
            points=' '.join(f"{50+800*(r['epoch']-lo)/max(hi-lo,1):.1f},{y+190-140*r[metric]:.1f}" for r in rs if r[metric] is not None)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="50" y="{y+212}" font-size="12">Epoch {lo}–{hi}; blue lift / red fall / green task; y=0–1</text>')
    parts.append('</svg>');(a.output/'curves.svg').write_text('\n'.join(parts))
    (a.output/'README.md').write_text('Existing evaluation aggregation; not a new training result.\n'
        'Legacy lift/held span 300 steps; fall/task are clip metrics. Auxiliary lift uses first stored reference z.\n'
        'End error is conditional on survival: always report survivors alongside it.\n'
        'First tracking failure is a >10cm error/alive threshold, not a causal failure diagnosis.\n'
        'Goal displacement uses first stored post-step position to last alive position; not path length.\n'
        'Stable lift is not full transport success. Wrist proximity is not measured contact.\n')
    print(f'{len(rows)} evaluations -> {a.output}')


if __name__=='__main__':main()
