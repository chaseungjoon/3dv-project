"""Checkpoint accounting shared by CPU fixtures and the GPU-machine audit entry point."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def check_span(start,end,source_epoch,target_epoch,num_envs=2048,horizon=32):
    if int(start['epoch'])!=source_epoch or int(end['epoch'])!=target_epoch:
        raise ValueError('Checkpoint epoch does not match requested training span')
    delta=int(end['frame'])-int(start['frame'])
    expected=(target_epoch-source_epoch)*num_envs*horizon
    if expected<=0 or delta!=expected:raise ValueError(f'Frame delta mismatch: {delta} != {expected}')
    return dict(source_epoch=source_epoch,final_epoch=target_epoch,source_frame=int(start['frame']),
                final_frame=int(end['frame']),extra_frames=delta)


def audit_files(source,checkpoint,source_epoch,target_epoch,output,source_sha):
    # Invoked in the team's prepared Python environment, never during CPU-only planning.
    import torch
    if digest(source)!=source_sha:raise ValueError('Source checkpoint changed')
    start=torch.load(source,map_location='cpu')
    end=torch.load(checkpoint,map_location='cpu')
    for payload in [start,end]:
        for key in ['model','optimizer','env_state','rng_state','running_mean_std']:
            if payload.get(key) is None:raise ValueError('Incomplete checkpoint: '+key)
    result=check_span(start,end,source_epoch,target_epoch)
    result.update(status='PASS',source_sha256=source_sha,checkpoint_sha256=digest(checkpoint),checkpoint=str(Path(checkpoint).resolve()))
    path=Path(output);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','checkpoint','source-sha','output']:p.add_argument('--'+key,required=True)
    for key in ['source-epoch','target-epoch']:p.add_argument('--'+key,type=int,required=True)
    a=p.parse_args()
    audit_files(a.source,a.checkpoint,a.source_epoch,a.target_epoch,a.output,a.source_sha)
