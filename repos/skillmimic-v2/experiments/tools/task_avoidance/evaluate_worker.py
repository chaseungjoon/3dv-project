"""Fresh-process evaluation or a donor-prefix capability probe. No PPO training."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from core import load_protocol, protocol_hash, assert_initial_equal, summarize, build_factor_function, FACTOR_NAMES, require_finite
from checkpoints import digest
from runtime import ROOT,build_player,cpu,snapshot


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','checkpoint-sha','protocol','output']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--reference-initial',type=Path)
    ap.add_argument('--prefix-checkpoint');ap.add_argument('--prefix-sha');ap.add_argument('--prefix-steps',type=int,default=0)
    ap.add_argument('--control-no-switch',action='store_true')
    a=ap.parse_args();p=load_protocol(a.protocol);out=Path(a.output)
    prefix=Path(a.prefix_checkpoint or a.checkpoint);prefix_sha=a.prefix_sha or a.checkpoint_sha
    if digest(a.checkpoint)!=a.checkpoint_sha or digest(prefix)!=prefix_sha:raise ValueError('Checkpoint changed')
    horizon=p['expected_clip_frames']-p['start_frame']
    if not 0<=a.prefix_steps<horizon:raise ValueError('Invalid prefix length')
    if a.control_no_switch and prefix_sha!=a.checkpoint_sha:raise ValueError('No-switch control must use donor checkpoint')
    out.mkdir(parents=True) # exclusive; preserves failures
    player,task,obs=build_player(prefix,p,a.seed)
    import torch
    import env.tasks.skillmimic_parahome as taskmod
    md=task._motion_data
    episode_initial=snapshot(task,obs)
    assert_initial_equal(episode_initial,episode_initial,p['initial_atol'])
    np.savez_compressed(out/'episode_initial.npz',**episode_initial)
    factors_fn=build_factor_function(ROOT/'skillmimic/env/tasks/skillmimic_parahome.py',vars(taskmod))
    original=taskmod.compute_humanoid_reward;latest={}
    def instrument(*args):
        reward=original(*args);factors=factors_fn(*args)
        if not torch.isfinite(factors).all() or not torch.allclose(reward,factors[:,-1],rtol=0,atol=p['reward_atol']):
            raise ValueError('Diagnostic factors differ from real reward')
        latest['factors']=factors.detach().clone();return reward
    taskmod.compute_humanoid_reward=instrument
    names=['reward','factors','obj','ref_obj','wrist_d','wrist_pos','done','physical_done','fall','train_stop','action']
    arrays={k:[] for k in names};prefix_data={k:[] for k in names}
    alive=np.ones(task.num_envs,bool);started=time.time();initial=None
    try:
        with torch.no_grad():
            for step in range(horizon):
                if step==a.prefix_steps:
                    # No state teleport: the SAME donor performed all previous actions in this simulator.
                    initial=snapshot(task,obs,step,alive,episode_initial['target'])
                    assert_initial_equal(initial,initial,p['initial_atol'])
                    if a.reference_initial:
                        with np.load(a.reference_initial,allow_pickle=False) as bank:assert_initial_equal(dict(bank),initial,p['initial_atol'])
                    np.savez_compressed(out/'initial.npz',**initial)
                    if not a.control_no_switch:player.restore(a.checkpoint)
                    # Check restore did not mutate simulator state/history/raw policy input.
                    assert_initial_equal(initial,snapshot(task,obs,step,alive,episode_initial['target']),p['initial_atol'])
                action=player.get_action(obs,True)
                raw,reward,done,info=player.env_step(player.env,action);obs=player.obs_to_torch(raw)
                obj=task._target_states[:,:3].clone();offset=6+2*task._dof_obs_size
                refo=task._curr_ref_obs[:,offset:offset+3]
                wrist=task._rigid_body_pos[:,[10,33],:]
                wd=(wrist-obj[:,None,:]).norm(dim=-1).min(-1).values
                pre_counter=task.progress_buf-1
                obj_stop=((task._curr_obs[:,offset:offset+3]-refo).norm(dim=-1)>task.obj_term_dist)&(pre_counter>1)
                if task.obj_term_dist<=0:obj_stop=torch.zeros_like(obj_stop)
                remain=md.motion_lengths[md.envid2motid]-md.envid2sframe
                clip_end=pre_counter>=remain-1
                # Fresh conditional 60-step window for the switched policy, never called the training objective.
                train_stop=obj_stop | (pre_counter>=torch.minimum(remain,torch.full_like(remain,a.prefix_steps+p['train_episode_length']))-1)
                factors=latest['factors']
                if not torch.allclose(reward.reshape(-1),factors[:,-1],rtol=0,atol=p['reward_atol']):raise ValueError('Delivered reward differs from base')
                record=dict(reward=reward.reshape(-1),factors=factors,obj=obj,ref_obj=refo,wrist_d=wd,wrist_pos=wrist,
                            done=done.reshape(-1).bool()|clip_end,physical_done=done.reshape(-1).bool(),
                            fall=info['terminate'].reshape(-1).bool(),train_stop=train_stop,action=action)
                record={k:cpu(v) for k,v in record.items()};require_finite(record)
                dest=prefix_data if step<a.prefix_steps else arrays
                for key,val in record.items():dest[key].append(val)
                alive &= ~record['done']
    finally:taskmod.compute_humanoid_reward=original
    arrays={k:np.stack(v) for k,v in arrays.items()};rows=summarize(arrays,initial,p)
    np.savez_compressed(out/'trajectory.npz',**arrays)
    if a.prefix_steps:np.savez_compressed(out/'prefix.npz',**{k:np.stack(v) for k,v in prefix_data.items()})
    payload=dict(status='COMPLETED',protocol_hash=protocol_hash(p),checkpoint=str(Path(a.checkpoint).resolve()),
                 seed=a.seed,checkpoint_sha256=a.checkpoint_sha,prefix_sha256=prefix_sha,prefix_steps=a.prefix_steps,
                 control_no_switch=a.control_no_switch,seconds=time.time()-started,control_dt_s=float(task.dt),
                 effective_env_cfg=task.cfg,factor_names=FACTOR_NAMES,episodes=rows,
                 initial_pairing='PASS' if a.reference_initial else 'REFERENCE',diagnostic_reward_match='PASS',
                 return_scope='finite observed conditional rollout; train_window is rule-censored prefix, no critic bootstrap')
    # task.cfg holds the argparse Namespace under 'args'; record non-JSON values as text.
    (out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False,default=str)+'\n')


if __name__=='__main__':main()
