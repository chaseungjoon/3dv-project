"""One checkpoint in one fresh process. GPU only; actual fix reward/path is retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
from core import load_protocol, protocol_hash, assert_initial_equal, summarize, build_factor_function, FACTOR_NAMES

ROOT=Path(__file__).resolve().parents[3]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--checkpoint-sha',required=True)
    ap.add_argument('--protocol',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--reference-initial',type=Path)
    a=ap.parse_args();p=load_protocol(a.protocol)
    if a.output.exists():raise ValueError('Output exists; refusing overwrite')
    if not a.checkpoint.is_file():raise ValueError('Checkpoint missing')
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=a.checkpoint_sha:
        raise ValueError('Checkpoint changed after planning')
    a.output.mkdir(parents=True)
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT/'experiments/tools'))
    sys.path.insert(0,str(ROOT/'skillmimic'))
    from isaacgym import gymapi  # must precede torch
    import torch
    from rl_games.torch_runner import Runner
    import methods as M
    import run as R
    def seed_all(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    seed_all(a.seed)
    horizon=p['expected_clip_frames']-p['start_frame']
    # Last transition reaches last valid reference frame. Timeout occurs at that transition.
    sys.argv=['skillmimic/run.py','--test']+M.base_args('place_book','fix')+[
        '--num_envs',str(p['num_envs']),'--episode_length',str(horizon),
        '--state_init',str(p['start_frame']),'--checkpoint',str(a.checkpoint),
        '--seed',str(a.seed),'--headless']
    captured={};old_run=Runner.run
    Runner.run=lambda self,args:captured.setdefault('runner',self)
    try:R.main()
    finally:Runner.run=old_run
    runner=captured['runner'];player=runner.create_player();task=player.env.task
    if player.is_rnn:raise ValueError('Recurrent policies need explicit hidden-state pairing')
    player.restore(str(a.checkpoint))
    md=task._motion_data
    if md.num_motions!=1 or int(md.motion_lengths[0])!=p['expected_clip_frames']:
        raise ValueError('Unexpected clip count/length')
    if p['gamma']!=float(runner.config['gamma']):raise ValueError('Gamma differs from training config')
    seed_all(a.seed)
    obs=player.env_reset();player.get_batch_size(obs['obs'],1)
    # Existing noisy initialization may reassign reference phase (STF); record and pair it, do not disable it.
    ref=md.hoi_data_dict[0]
    def cpu(t):return t.detach().cpu().numpy().copy()
    initial={k:cpu(getattr(task,v)) for k,v in {
        'actors':'_root_states','dofs':'_dof_state','rigid_bodies':'_rigid_body_state',
        'contact_forces':'_contact_forces','target':'_target_states',
        'history':'_hist_obs_batch','previous_hoi':'_hist_obs','current_hoi':'_curr_obs',
        'raw_observation':'obs_buf','reference':'_curr_ref_obs','progress':'progress_buf',
        'buffer':'buffer_steps'}.items()}
    initial.update(policy_input=cpu(obs['obs']),start_frames=cpu(md.envid2sframe),
                   motion_ids=cpu(md.envid2motid),clip_steps=cpu(md.motion_lengths[md.envid2motid]-md.envid2sframe),goal=cpu(ref['obj_pos'][-1]),
                   reference_lift=np.asarray(float((ref['obj_pos'][:,2]-ref['obj_pos'][0,2]).max())))
    initial.update({'reward_weight_'+key:cpu(value) for key,value in md.reward_weights.items()})
    assert_initial_equal(initial,initial,p['initial_atol']) # includes finite check
    if a.reference_initial:
        with np.load(a.reference_initial,allow_pickle=False) as bank:
            assert_initial_equal(dict(bank),initial,p['initial_atol'])
    np.savez_compressed(a.output/'initial.npz',**initial)

    import env.tasks.skillmimic_parahome as taskmod
    source=ROOT/'skillmimic/env/tasks/skillmimic_parahome.py'
    factors_fn=build_factor_function(source,vars(taskmod));original=taskmod.compute_humanoid_reward
    latest={}
    def instrument(*args):
        reward=original(*args)
        with torch.no_grad():
            factors=factors_fn(*args)
            if not torch.isfinite(factors).all() or not torch.allclose(reward,factors[:,-1],rtol=0,atol=p['reward_atol']):
                raise ValueError('Diagnostic factor reconstruction differs from real reward')
            latest['factors']=factors.detach().clone()
        return reward
    taskmod.compute_humanoid_reward=instrument
    arrays={k:[] for k in ['reward','factors','obj','ref_obj','wrist_d','done','physical_done','fall','train_stop','action']}
    started=time.time()
    try:
        with torch.no_grad():
            for step in range(horizon):
                action=player.get_action(obs,True)
                raw,reward,done,info=player.env_step(player.env,action)
                obs=player.obs_to_torch(raw)
                obj=task._target_states[:,:3].clone()
                offset=6+2*task._dof_obs_size
                refo=task._curr_ref_obs[:,offset:offset+3]
                wd=(task._rigid_body_pos[:,[10,33],:]-obj[:,None,:]).norm(dim=-1).min(-1).values
                # Use EXACT training object-error inputs/predicate, including its current HOI timing.
                obj_stop=((task._curr_obs[:,offset:offset+3]-refo).norm(dim=-1)>task.obj_term_dist)&((task.progress_buf-1)>1)
                if task.obj_term_dist<=0:obj_stop=torch.zeros_like(obj_stop)
                # Reset checks use the pre-increment counter (post_physics_step increments it AFTER reset checks).
                train_len=torch.minimum(md.motion_lengths[md.envid2motid]-md.envid2sframe,
                                        torch.full_like(md.envid2sframe,p['train_episode_length']))
                clip_end=(task.progress_buf-1)>=(md.motion_lengths[md.envid2motid]-md.envid2sframe-1)
                train_stop=obj_stop | ((task.progress_buf-1)>=train_len-1)
                factors=latest['factors']
                if not torch.allclose(reward.reshape(-1),factors[:,-1],rtol=0,atol=p['reward_atol']):
                    raise ValueError('Delivered reward differs from base reward; buffer/override needs review')
                for key,val in dict(reward=reward.reshape(-1),factors=factors,obj=obj,ref_obj=refo,
                                    wrist_d=wd,done=done.reshape(-1).bool()|clip_end,physical_done=done.reshape(-1).bool(),fall=info['terminate'].reshape(-1).bool(),
                                    train_stop=train_stop,action=action).items():arrays[key].append(cpu(val))
    finally:taskmod.compute_humanoid_reward=original
    arrays={k:np.stack(v) for k,v in arrays.items()}
    rows=summarize(arrays,initial,p)
    np.savez_compressed(a.output/'trajectory.npz',**arrays)
    payload=dict(status='COMPLETED',protocol_hash=protocol_hash(p),checkpoint=str(a.checkpoint),
                 seed=a.seed,checkpoint_sha256=a.checkpoint_sha,seconds=time.time()-started,control_dt_s=float(task.dt),
                 effective_env_cfg=task.cfg,factor_names=FACTOR_NAMES,episodes=rows,
                 initial_pairing='PASS' if a.reference_initial else 'REFERENCE',
                 diagnostic_reward_match='PASS',
                 return_scope='finite observed rollout; train_window is training-rule-censored prefix, no value bootstrap; not expected training objective')
    # cfg is normally JSON-safe; explicitly fail rather than silently stringify tensors.
    (a.output/'result.json').write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':main()
