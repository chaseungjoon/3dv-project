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
    if p['task'] != 'place_book' or p['version'] not in ['avoidance_v1', 'avoidance_v2']:
        raise ValueError('Only avoidance_v1/v2 Place Book is supported')
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
    lift_key = 'partial_lift_m' if p['version'] == 'avoidance_v2' else 'lift_attempt_m'
    for key in ['gamma', 'wrist_near_m', lift_key, 'lift_success_ref_fraction',
                'tracking_error_m', 'progress_clip']:
        if not np.isfinite(p[key]) or p[key] <= 0: raise ValueError('Invalid '+key)
    if p['gamma'] > 1: raise ValueError('gamma exceeds 1')
    for key in ['initial_atol', 'reward_atol', 'repeat_position_atol_m', 'repeat_reward_atol', 'progress_weight']:
        if not np.isfinite(p[key]) or p[key] < 0: raise ValueError('Invalid '+key)
    if p['version'] == 'avoidance_v2':
        for key in ['reach_window_steps', 'reach_hold_steps']:
            if type(p[key]) is not int or p[key] <= 0: raise ValueError('Invalid '+key)
        if p['reach_hold_steps'] > p['reach_window_steps']+1:
            raise ValueError('Reach hold exceeds measurement window')
        for key in ['reach_closing_m', 'reach_wrist_motion_m']:
            if not np.isfinite(p[key]) or p[key] <= 0: raise ValueError('Invalid '+key)
        steps = p['capability_prefix_steps']
        horizon = p['expected_clip_frames'] - p['start_frame']
        if not steps or len(set(steps)) != len(steps) or any(type(x) is not int or not 0 <= x < horizon for x in steps):
            raise ValueError('Invalid capability prefix steps')
    return p


def protocol_hash(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_finite(arrays):
    for name, value in arrays.items():
        a = np.asarray(value)
        if not np.issubdtype(a.dtype, np.number) and a.dtype != bool:
            raise ValueError('Nonnumeric array: '+name)
        if not np.isfinite(a).all(): raise ValueError('Nonfinite array: '+name)


def unique_state_ids(initial, n):
    """Exact duplicate fingerprints in environment-local actor coordinates; not independent-run IDs."""
    keys = ['local_actor_state', 'dof_pos', 'dof_vel', 'history', 'policy_input', 'start_frames', 'progress']
    if any(k not in initial or np.asarray(initial[k]).shape[0] != n for k in keys):
        return [None] * n
    ids = []
    for e in range(n):
        h = hashlib.sha256()
        for k in keys:
            a = np.asarray(initial[k][e]);h.update(k.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        ids.append(h.hexdigest())
    return ids


def reach_events(z, valid, p):
    """Proxy requiring wrist motion TOWARD a fixed prior object position, independent of lift.

    The object approaching a stationary wrist cannot trigger this predicate. The same wrist
    must satisfy the window displacement and sustained proximity conditions.
    """
    wrist = np.asarray(z['wrist_pos']); obj = np.asarray(z['obj'])
    if wrist.shape != obj.shape[:2] + (2, 3): raise ValueError('Invalid wrist_pos shape')
    events = np.zeros(valid.shape, bool)
    window, hold = p['reach_window_steps'], p['reach_hold_steps']
    for t in range(window, len(obj)):
        lo = t-window
        old = np.linalg.norm(wrist[lo]-obj[lo, :, None], axis=-1)
        new = np.linalg.norm(wrist[t]-obj[lo, :, None], axis=-1)
        motion = np.linalg.norm(wrist[t]-wrist[lo], axis=-1)
        start = max(lo, t-hold+1)
        if t-start+1 < hold: continue
        near = (np.linalg.norm(wrist[start:t+1]-obj[start:t+1, :, None], axis=-1) < p['wrist_near_m']).all(0)
        active = valid[lo:t+1].all(0)
        events[t] = active & ((old-new >= p['reach_closing_m']) &
                             (motion >= p['reach_wrist_motion_m']) & near).any(-1)
    return events


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
    if done.shape != train_stop.shape or done.ndim != 2 or not all(done.shape): raise ValueError('Invalid mask shapes')
    def mask(stop):
        return np.concatenate([np.ones((1,stop.shape[1]),bool),~np.maximum.accumulate(stop,axis=0)[:-1]],axis=0)
    return mask(done),mask(done|train_stop)


def summarize(z, initial, p):
    require_finite(z);require_finite(initial)
    reward=z['reward'];valid,train_valid=transition_masks(z['done'],z['train_stop'])
    if not np.isfinite(reward).all(): raise ValueError('Nonfinite reward')
    T,N=reward.shape
    offset = int(initial.get('segment_start_step', 0))
    if T != p['expected_clip_frames']-p['start_frame']-offset: raise ValueError('Unexpected evaluation horizon')
    if N != p['num_envs']: raise ValueError('Unexpected environment count')
    for key in ['done','fall','train_stop','wrist_d']:
        if z[key].shape != (T,N): raise ValueError('Invalid shape: '+key)
    for key in ['obj','ref_obj']:
        if z[key].shape != (T,N,3): raise ValueError('Invalid shape: '+key)
    if z['factors'].shape != (T,N,len(FACTOR_NAMES)): raise ValueError('Invalid factor shape')
    eligible = np.asarray(initial.get('eligible',np.ones(N)),dtype=bool)
    valid &= eligible[None];train_valid &= eligible[None]
    discounts=p['gamma']**np.arange(T)
    errors=np.linalg.norm(z['obj']-z['ref_obj'],axis=-1)
    near=z['wrist_d'] < p['wrist_near_m']
    dz=z['obj'][:,:,2]-initial['target'][:,2]
    safe=valid & ~z['fall'].astype(bool) & ~z['done'].astype(bool)
    clip_steps=initial.get('clip_steps',np.full(N,T)).astype(int)
    if np.any(clip_steps[eligible]<1) or np.any(clip_steps[eligible]>T):raise ValueError('Clip window outside recorded horizon')
    # Only the last valid reference transition may be a non-fall timeout.
    for e,end in enumerate(clip_steps-1):
        if eligible[e]:safe[end,e]=valid[end,e] and not bool(z['fall'][end,e])
    partial_threshold = p.get('partial_lift_m',p.get('lift_attempt_m'))
    attempt=near & (dz>=partial_threshold) & safe
    lift=near & (dz>=float(initial['reference_lift'])*p['lift_success_ref_fraction']) & safe
    reaching = reach_events(z,safe,p) if p['version']=='avoidance_v2' else None
    state_ids = unique_state_ids(initial,N)
    rows=[]
    for e in range(N):
        if not eligible[e]:
            rows.append(dict(env_id=e,eligible=0,exclusion='donor terminated or reference ended before switch',state_id=state_ids[e]))
            continue
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
        if p['version']=='avoidance_v2':
            row=rows[-1]
            row.update(eligible=1,exclusion=None,state_id=state_ids[e],segment_start_step=offset,
                       partial_lift=row.pop('lift_attempt'),first_partial_lift_step=row.pop('first_attempt_step'))
            attempts=np.flatnonzero(reaching[:,e])
            row.update(reach_attempt_proxy=int(len(attempts)>0),first_reach_step=int(attempts[0]+1) if len(attempts) else None,
                       reach_without_partial_lift=int(len(attempts)>0 and not attempt[:,e].any()))
            origin=np.asarray(initial.get('origin_target',initial['target']))
            raised_at_start=initial['target'][e,2]-origin[e,2] >= partial_threshold
            start_near=float(initial.get('wrist_d',np.full(N,np.inf))[e]) < p['wrist_near_m']
            row['starting_stage']='initial' if offset==0 else 'holding' if raised_at_start and start_near else 'near_object' if start_near else 'approach'
            still_raised=(z['obj'][:,e,2]-origin[e,2]>=partial_threshold)&near[:,e]&safe[:,e]
            row['maintained_initial_lift']=int(complete and still_raised[v].all()) if raised_at_start and start_near else None
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
