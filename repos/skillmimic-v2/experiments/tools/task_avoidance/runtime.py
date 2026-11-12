"""Lazy GPU runtime; importing this module on the CPU orchestrator is safe."""
import os
from pathlib import Path
import random
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def build_player(checkpoint,p,seed,method='fix',training_path=False):
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT/'experiments/tools'));sys.path.insert(0,str(ROOT/'skillmimic'))
    from isaacgym import gymapi  # before torch
    import torch
    from rl_games.torch_runner import Runner
    import methods as M
    import run as R
    seed_all(seed)
    horizon=p['train_episode_length'] if training_path else p['expected_clip_frames']-p['start_frame']
    sys.argv=['skillmimic/run.py']+([] if training_path else ['--test'])+M.base_args('place_book',method)+[
        '--num_envs',str(p['num_envs']),'--episode_length',str(horizon),
        '--state_init',str(p['start_frame']),'--checkpoint',str(checkpoint),'--seed',str(seed),'--headless']
    if training_path:sys.argv+=['--enable_buffernode','--reweight','--reweight_alpha','1.0']
    captured={};old=Runner.run
    Runner.run=lambda self,args:captured.setdefault('runner',self)
    try:R.main()
    finally:Runner.run=old
    runner=captured['runner'];player=runner.create_player();task=player.env.task
    if player.is_rnn:raise ValueError('Recurrent policies need explicit hidden-state pairing')
    player.restore(str(checkpoint))
    md=task._motion_data
    if md.num_motions!=1 or int(md.motion_lengths[0])!=p['expected_clip_frames']:raise ValueError('Unexpected motion')
    if p['gamma']!=float(runner.config['gamma']):raise ValueError('Gamma differs from training config')
    seed_all(seed);obs=player.env_reset();player.get_batch_size(obs['obs'],1)
    return player,task,obs


def cpu(t):return t.detach().cpu().numpy().copy()


def snapshot(task,obs,offset=0,eligible=None,origin_target=None):
    md=task._motion_data;ref=md.hoi_data_dict[0];n=task.num_envs
    initial={k:cpu(getattr(task,v)) for k,v in {
        'actors':'_root_states','dofs':'_dof_state','dof_pos':'_dof_pos','dof_vel':'_dof_vel',
        'rigid_bodies':'_rigid_body_state','contact_forces':'_contact_forces','target':'_target_states',
        'history':'_hist_obs_batch','previous_hoi':'_hist_obs','current_hoi':'_curr_obs',
        'raw_observation':'obs_buf','reference':'_curr_ref_obs','progress':'progress_buf','buffer':'buffer_steps'}.items()}
    actor=initial['actors'].reshape(n,-1,13).copy()
    actor[:,:,:2]-=actor[:,0:1,:2].copy()
    initial.update(local_actor_state=actor,policy_input=cpu(obs['obs']),start_frames=cpu(md.envid2sframe),
                   motion_ids=cpu(md.envid2motid),clip_steps=cpu(md.motion_lengths[md.envid2motid]-md.envid2sframe)-offset,
                   goal=cpu(ref['obj_pos'][-1]),reference_lift=np.asarray(float((ref['obj_pos'][:,2]-ref['obj_pos'][0,2]).max())),
                   segment_start_step=np.asarray(offset),eligible=np.ones(n,bool) if eligible is None else np.array(eligible,bool,copy=True),
                   wrist_d=cpu((task._rigid_body_pos[:,[10,33],:]-task._target_states[:,None,:3]).norm(dim=-1).min(-1).values))
    initial['origin_target']=initial['target'].copy() if origin_target is None else origin_target.copy()
    initial.update({'reward_weight_'+key:cpu(value) for key,value in md.reward_weights.items()})
    return initial
