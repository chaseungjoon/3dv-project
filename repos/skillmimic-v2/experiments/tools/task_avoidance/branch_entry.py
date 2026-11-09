"""GPU entry point for explicit C0/C1 branching. Original source checkpoint is read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from checkpoints import audit_files

ROOT=Path(__file__).resolve().parents[3]


def main():
    idx=sys.argv.index('--');ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--seed',required=True,type=int)
    ap.add_argument('--source-sha',required=True)
    ap.add_argument('--source-epoch',required=True,type=int)
    ap.add_argument('--report',required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(sys.argv[1:idx]);run_args=sys.argv[idx+1:]
    sys.path.insert(0,str(ROOT/'skillmimic'));os.chdir(ROOT)
    from isaacgym import gymapi
    import torch
    import numpy as np
    from learning.skillmimic_agent import SkillMimicAgent
    from rl_games.algos_torch import torch_ext
    import run as R
    original=SkillMimicAgent.restore_full_state
    def restore(self,filename):
        h=hashlib.sha256(Path(a.source).read_bytes()).hexdigest()
        if h!=a.source_sha:raise ValueError('Source checkpoint hash changed')
        checkpoint=torch_ext.load_checkpoint(filename)
        required=['model','optimizer','epoch','frame','env_state','rng_state','running_mean_std']
        if any(k not in checkpoint or checkpoint[k] is None for k in required):
            raise ValueError('Full checkpoint state missing; refusing partial-state branch')
        if not a.resume and int(checkpoint['epoch'])!=a.source_epoch:raise ValueError('Unexpected source epoch')
        target=int(run_args[run_args.index('--max_epochs')+1])+1
        if int(checkpoint['epoch'])>=target:
            raise ValueError('Already at/past target: audit checkpoint instead of training')
        original(self,filename)
        # New branch replicate starts from SAME optimizer/normalizer/ATS, but paired NEW RNG stream.
        # An interrupted branch resume retains its saved RNG instead.
        if not a.resume:
            random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        record=dict(source=a.source,source_sha256=h,loaded_checkpoint=filename,
                    restored_epoch=int(self.epoch_num),restored_frame=int(self.frame),
                    state_keys=list(checkpoint),branch_seed=a.seed,reseeded=not a.resume,
                    physical_state='fresh episode reset; source file contains ATS state, not full simulator/contact state',
                    status='RESTORED')
        with open(a.report,'a') as f:f.write(json.dumps(record)+'\n')
    SkillMimicAgent.restore_full_state=restore
    try:
        sys.argv=['skillmimic/run.py']+run_args
        R.main()
        target=int(run_args[run_args.index('--max_epochs')+1])+1
        folder=Path(run_args[run_args.index('--output_path')+1])/run_args[run_args.index('--experiment')+1]
        files=list((folder/'nn').glob(f'*_{target:08d}.pth'))
        if len(files)!=1:raise ValueError('Expected final numbered checkpoint missing/ambiguous')
        audit_files(a.source,str(files[0]),a.source_epoch,target,folder/'completion.json',a.source_sha)
    finally:SkillMimicAgent.restore_full_state=original


if __name__=='__main__':main()
