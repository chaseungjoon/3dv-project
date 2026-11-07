"""CPU metric and pairing logic. No simulator dependencies."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

FACTOR_NAMES = ['rb', 'ro', 'rig', 'rp', 'rr', 'rrv', 'r_vel_diff', 'rop', 'ror', 'ropv', 'reward']


def load_protocol(path):
    p = json.loads(Path(path).read_text())
    if p['task'] != 'place_book' or p['version'] != 'avoidance_v1':
        raise ValueError('Only avoidance_v1 Place Book is supported')
    for key in ['epochs', 'eval_seeds', 'branch_seeds']:
        if not p[key] or len(set(p[key])) != len(p[key]) or any(type(x) is not int or x < 0 for x in p[key]):
            raise ValueError('Invalid unique nonnegative integers: '+key)
    if len(p['primary_pair']) != 2 or not set(p['primary_pair']) <= set(p['epochs']):
        raise ValueError('Primary pair must be two candidate epochs')
    if p['primary_pair'][0] >= p['primary_pair'][1]:
        raise ValueError('Primary pair must be ordered early/late')
    for key in ['num_envs', 'expected_clip_frames', 'train_episode_length', 'branch_extra_epochs']:
        if type(p[key]) is not int or p[key] <= 0: raise ValueError('Invalid '+key)
    if not 0 <= p['start_frame'] < p['expected_clip_frames']-1: raise ValueError('Invalid start frame')
    for key in ['gamma', 'wrist_near_m', 'lift_attempt_m', 'lift_success_ref_fraction',
                'tracking_error_m', 'progress_clip']:
        if not np.isfinite(p[key]) or p[key] <= 0: raise ValueError('Invalid '+key)
    if p['gamma'] > 1: raise ValueError('gamma exceeds 1')
    for key in ['initial_atol', 'reward_atol', 'repeat_position_atol_m', 'repeat_reward_atol', 'progress_weight']:
        if not np.isfinite(p[key]) or p[key] < 0: raise ValueError('Invalid '+key)
    return p


def protocol_hash(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def assert_initial_equal(a, b, atol=0):
    if set(a) != set(b): raise ValueError('Initial state keys differ')
    for key in a:
        x,y=np.asarray(a[key]),np.asarray(b[key])
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Initial state invalid: '+key)
        if not np.allclose(x,y,rtol=0,atol=atol):
            raise ValueError('Initial state mismatch: '+key)


def transition_masks(done, train_stop):
    """Include terminal transition; exclude ALL later transitions, even if simulator keeps running."""
    done=np.asarray(done,dtype=bool);train_stop=np.asarray(train_stop,dtype=bool)
    if done.shape != train_stop.shape: raise ValueError('mask shapes differ')
    def mask(stop):
        return np.concatenate([np.ones((1,stop.shape[1]),bool),~np.maximum.accumulate(stop,axis=0)[:-1]],axis=0)
    return mask(done),mask(done|train_stop)


def summarize(z, initial, p):
    reward=z['reward'];valid,train_valid=transition_masks(z['done'],z['train_stop'])
    if not np.isfinite(reward).all(): raise ValueError('Nonfinite reward')
    T,N=reward.shape
    if T != p['expected_clip_frames']-p['start_frame']: raise ValueError('Unexpected evaluation horizon')
    discounts=p['gamma']**np.arange(T)
    errors=np.linalg.norm(z['obj']-z['ref_obj'],axis=-1)
    near=z['wrist_d'] < p['wrist_near_m']
    dz=z['obj'][:,:,2]-initial['target'][:,2]
    safe=valid & ~z['fall'].astype(bool) & ~z['done'].astype(bool)
    clip_steps=initial.get('clip_steps',np.full(N,T)).astype(int)
    if np.any(clip_steps<1) or np.any(clip_steps>T):raise ValueError('Clip window outside recorded horizon')
    # Only the last valid reference transition may be a non-fall timeout.
    for e,end in enumerate(clip_steps-1):
        safe[end,e]=valid[end,e] and not bool(z['fall'][end,e])
    attempt=near & (dz>=p['lift_attempt_m']) & safe
    lift=near & (dz>=float(initial['reference_lift'])*p['lift_success_ref_fraction']) & safe
    rows=[]
    for e in range(N):
        v=valid[:,e];tv=train_valid[:,e];last=np.flatnonzero(v)[-1]
        fall=bool((z['fall'][:,e].astype(bool)&v).any())
        end=clip_steps[e]-1
        complete=bool(v[:end+1].all() and not fall and not z['done'][:end,e].any())
        goal=initial['goal'];d0=np.linalg.norm(initial['target'][e,:3]-goal)
        progress=d0-np.linalg.norm(z['obj'][last,e]-goal)
        first=np.flatnonzero(attempt[:,e]);bad=np.flatnonzero((errors[:end+1,e]>p['tracking_error_m'])|~safe[:end+1,e])
        rows.append(dict(env_id=e,reference_steps=int(clip_steps[e]),
                         actual_start_frame=int(initial.get('start_frames',np.full(N,p['start_frame']))[e]),return_eval=float((reward[:,e]*v*discounts).sum()),
                         return_train_window=float((reward[:,e]*tv*discounts).sum()),
                         undiscounted_eval=float((reward[:,e]*v).sum()),
                         observed_steps=int(v.sum()),train_window_steps=int(tv.sum()),
                         lift_attempt=int(attempt[:,e].any()),lift_success=int(lift[:,e].any()),
                         first_attempt_step=int(first[0]+1) if len(first) else None,
                         goal_progress_m=float(progress),fall=int(fall),
                         stable_lift=int(lift[:,e].any() and complete),
                         task_success=int(complete and (errors[:end+1,e]<=p['tracking_error_m']).all()),
                         wrist_min_m=float(z['wrist_d'][v,e].min()),
                         wrist_near_steps=int((near[:,e]&v).sum()),
                         held_steps=int(attempt[:,e].sum()),
                         first_tracking_failure_step=int(bad[0]+1) if len(bad) else None,
                         goal_error_survivors_m=float(np.linalg.norm(z['obj'][end,e]-goal)) if complete else None,
                         **{'factor_'+name:float(z['factors'][v,e,i].mean()) for i,name in enumerate(FACTOR_NAMES)}))
    return rows


def build_factor_function(source_path, namespace):
    """Diagnostic-only AST copy of the real reward function; never replaces its returned reward.

    Keep all arithmetic, change only final return into a factor stack. Compare reconstructed
    reward to the original JIT function at runtime. Fail closed if the source structure changes.
    """
    tree=ast.parse(Path(source_path).read_text())
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_humanoid_reward'))
    if not isinstance(fn.body[-1],ast.Return) or not isinstance(fn.body[-1].value,ast.Name) or fn.body[-1].value.id!='reward':
        raise ValueError('Reward return structure changed; review instrumentation')
    fn.name='_diagnostic_factors';fn.decorator_list=[];fn.returns=None
    for a in fn.args.args:a.annotation=None
    fn.body[-1]=ast.parse('return torch.stack(['+','.join(FACTOR_NAMES)+'], dim=-1)').body[0]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    scope=dict(namespace);exec(compile(module,str(source_path),'exec'),scope)
    return scope[fn.name]
