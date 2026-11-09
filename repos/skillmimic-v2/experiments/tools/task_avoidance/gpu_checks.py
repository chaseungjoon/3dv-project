"""Actual task-path checks on the team's GPU. No optimizer updates or policy training."""
import argparse
import json
import os
from pathlib import Path
from checkpoints import digest
from core import load_protocol,protocol_hash
from runtime import build_player


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','checkpoint-sha','protocol','output']:ap.add_argument('--'+key,required=True)
    a=ap.parse_args();p=load_protocol(a.protocol);out=Path(a.output);out.mkdir(parents=True)
    if digest(a.checkpoint)!=a.checkpoint_sha:raise ValueError('Checkpoint changed')
    os.environ['SMV2_PROGRESS_WEIGHT']='0';os.environ['SMV2_PROGRESS_CLIP']=str(p['progress_clip'])
    os.environ.pop('SMV2_PROGRESS_LOG',None)
    report=dict(status='NOT RUN',protocol_hash=protocol_hash(p),checkpoint_sha256=a.checkpoint_sha,checks={})
    try:
        player,task,obs=build_player(a.checkpoint,p,p['eval_seeds'][0],method='fix_progress',training_path=True)
        import torch
        from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRISBuffernodePhase as Parent
        if task.num_envs<2:raise ValueError('At least two environments required')
        inherited='_compute_reward' not in Parent.__dict__;original=Parent._compute_reward;captured={}
        def capture(self):
            captured['buffer']=self.buffer_steps.clone()>0
            original(self);captured['base']=self.rew_buf.clone()
        Parent._compute_reward=capture
        checks=dict(zero_weight_steps=0,progress_steps=0,buffer_transitions=0,terminal_transitions=0,partial_resets=0)
        done=torch.zeros(task.num_envs,dtype=torch.bool,device=task.device)
        try:
            with torch.no_grad():
                for step in range(p['train_episode_length']+10):
                    ids=done.nonzero(as_tuple=False).view(-1)
                    if len(ids):obs=player.env_reset(ids)
                    if step==5:task.buffer_steps[:]=2  # explicit fixture, actual parent buffer path
                    if step==8:
                        rest=torch.arange(1,task.num_envs,device=task.device)
                        history=task._hist_obs_batch[rest].clone();target=task._target_states[rest].clone()
                        obs=player.env_reset(torch.tensor([0],device=task.device))
                        if not torch.equal(history,task._hist_obs_batch[rest]) or not torch.equal(target,task._target_states[rest]):
                            raise ValueError('Partial reset changed untouched environment history/object state')
                        checks['partial_resets']+=1
                    task._progress_weight=0.0 if step<4 else 1.0
                    before=task._target_states[:,:3].clone()
                    action=player.get_action(obs,True)
                    raw,reward,done,info=player.env_step(player.env,action);obs=player.obs_to_torch(raw)
                    if not torch.equal(before,task._progress_before):raise ValueError('Progress cached a pre-reset state')
                    base=captured['base'];after=task._target_states[:,:3]
                    delta=((before-task._progress_goal).norm(dim=-1)-(after-task._progress_goal).norm(dim=-1))/task._progress_length
                    extra=delta.clamp(-task._progress_clip,task._progress_clip)*task._progress_weight
                    extra=torch.where(captured['buffer'],torch.zeros_like(extra),extra)
                    if not torch.isfinite(reward).all() or not torch.allclose(reward.reshape(-1),base+extra,rtol=0,atol=p['reward_atol']):
                        raise ValueError('Actual reward differs from parent+progress accounting')
                    if task._progress_weight==0:
                        if not torch.equal(reward.reshape(-1),base):raise ValueError('Weight zero changed parent reward')
                        checks['zero_weight_steps']+=1
                    else:checks['progress_steps']+=1
                    checks['buffer_transitions']+=int(captured['buffer'].sum())
                    checks['terminal_transitions']+=int(done.sum())
        finally:
            if inherited:del Parent._compute_reward
            else:Parent._compute_reward=original
        report['checks']=checks
        report['status']='PASS' if all(v>0 for v in checks.values()) else 'INCONCLUSIVE'
        report['scope']='Reward accounting and partial reset on real training task path; buffer counters explicitly set for coverage. Not physics validation or a learning result.'
    except BaseException as error:
        report.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['status']=='PASS' else 3


if __name__=='__main__':
    import sys
    sys.exit(main())
