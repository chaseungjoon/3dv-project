"""U0 instrumentation: which simulator state and which reference frame each part of a control step reads, plus the
effective (loaded) environment configuration. (GPU, ~1 min; changes nothing.)

    .venv/bin/python environment_iteration/src/probes/state_timing.py fix-6                 # fix-6 policy, own env
    .venv/bin/python environment_iteration/src/probes/state_timing.py fix-6 --env fix-8     # fix-6 policy in fix-8
        [--epoch 3000] [--num_envs 8] [--seed 101] [--horizon 150] [--out <dir>]

Rolls out the policy from the protocol's clean frame-2 start (stoch, --test like the evaluation) and after every
control step t (t = 0 .. horizon-1; physical step id = t + 1 = number of actions simulated) records, per env:

  label            envid2sframe + progress_buf (rollout.py `ref_frame`, the video HUD number)
  trace ref frame  motion-file row that `_curr_ref_obs` equals (what rollout.py stores as ref_obj / ref_key_pos)
  reward ref frame motion-file row of the reference the reward used (`_sync_ref_obs` with stateSync, else
                   `_curr_ref_obs`)
  reward state     whether the reward input `_curr_obs` (book, root) equals the refreshed state of this step
                   (s_{t+1}) or of the previous step (s_t), in metres
  phase frame      phase observation x clip length (last obs entry)
  history row      max |last history row - row rebuilt from the refreshed state| (the row appended this step)

Rows are matched to the motion file exactly (min over frames of the max abs difference; > 1e-4 = no exact match, -1).
Writes <out>/state_timing.json (manifest + per-step medians) and prints a table of the first steps and a summary.
Default out: iterations/<env>/probes/state_timing_<policy>_e<epoch>/.
"""
import argparse
import json
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

OBJ = slice(366, 369)


def manifest(task, it, env_it):
    """Loaded values, not the declared ones."""
    gym, env0 = task.gym, task.envs[0]
    sp = gym.get_sim_params(task.sim)
    px = sp.physx
    out = dict(
        policy_iteration=it, env_iteration=env_it, task_class=type(task).__name__,
        task_mro=[c.__name__ for c in type(task).__mro__[:8]],
        asset_file=task.cfg["env"]["asset"]["assetFileName"],
        termination_height_configured=task.cfg["env"].get("terminationHeight"),
        termination_height_effective=float(task._termination_heights),
        early_termination=bool(task._enable_early_termination),
        obj_termination_dist=float(getattr(task, "obj_term_dist", 0.0)),
        state_sync=bool(getattr(task, "_sync", False)),
        is_test=bool(task.isTest), max_episode_length=int(task.max_episode_length),
        obs_size=int(task.obs_buf.shape[-1]), ref_hoi_obs_size=int(task.ref_hoi_obs_size),
        history_length=int(task.history_length), hist_layout_fix=bool(task.cfg["env"].get("histLayoutFix", False)),
        reward_weights={k: float(v) for k, v in task.cfg["env"]["rewardWeights"].items()},
        sim=dict(dt=sp.dt, substeps=sp.substeps, solver_type=px.solver_type,
                 num_position_iterations=px.num_position_iterations,
                 num_velocity_iterations=px.num_velocity_iterations, contact_offset=px.contact_offset,
                 rest_offset=px.rest_offset, bounce_threshold_velocity=px.bounce_threshold_velocity,
                 max_depenetration_velocity=px.max_depenetration_velocity),
        actors=[gym.get_actor_name(env0, i) for i in range(gym.get_actor_count(env0))],
    )
    h = task._target_handles[0]
    bp = gym.get_actor_rigid_body_properties(env0, h)[0]
    sh = gym.get_actor_rigid_shape_properties(env0, h)
    out["book"] = dict(
        mass=float(bp.mass), com=[bp.com.x, bp.com.y, bp.com.z],
        inertia_diag=[bp.inertia.x.x, bp.inertia.y.y, bp.inertia.z.z], n_shapes=len(sh),
        shapes=[dict(friction=s.friction, rolling_friction=s.rolling_friction, torsion_friction=s.torsion_friction,
                     restitution=s.restitution, contact_offset=s.contact_offset, rest_offset=s.rest_offset)
                for s in sh])
    hh = task.humanoid_handles[0]
    hb = gym.get_actor_rigid_body_properties(env0, hh)
    hs = gym.get_actor_rigid_shape_properties(env0, hh)
    names = gym.get_actor_rigid_body_names(env0, hh)
    ranges = gym.get_actor_rigid_body_shape_indices(env0, hh)
    tip = names.index("jLeftSecondDIP")
    ts = hs[ranges[tip].start]

    def span(vals):
        return [float(min(vals)), float(max(vals))]
    out["humanoid"] = dict(
        n_bodies=len(hb), total_mass=float(sum(b.mass for b in hb)), n_shapes=len(hs),
        friction=span([s.friction for s in hs]), contact_offset=span([s.contact_offset for s in hs]),
        rest_offset=span([s.rest_offset for s in hs]),
        fingertip_jLeftSecondDIP=dict(friction=ts.friction, contact_offset=ts.contact_offset,
                                      rest_offset=ts.rest_offset, mass=float(hb[tip].mass)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration", help="whose policy")
    p.add_argument("--env", default="", help="whose environment (default: the policy's)")
    p.add_argument("--epoch", type=int, default=3000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--out", default="")
    a = p.parse_args()
    import numpy as np
    import iterlib as L
    import rollout as RO
    env_it = a.env or a.iteration
    ck = L.checkpoint_at(L.paths(a.iteration)["run_dir"], a.epoch)
    if not ck:
        sys.exit(f"no checkpoint for {a.iteration} epoch {a.epoch}")
    ov = {"cfg_env": L.load(env_it)["cfg_env"], "task": L.load(env_it)["task"]}
    env = RO.Env(a.iteration, a.num_envs, "stoch", 300, seed=a.seed, first_checkpoint=ck, overrides=ov)
    torch, task, player, md = env.torch, env.task, env.player, env.md
    man = manifest(task, a.iteration, env_it)
    H = md.hoi_data_dict[0]["hoi_data"].float()                     # (clip_frames, 428)
    Lc = int(md.motion_lengths[0])

    def match(rows):
        d = (rows.float()[:, None, :] - H[None]).abs().max(dim=-1).values    # (E, Lc), exact (no matmul trick)
        mn, ix = d.min(dim=1)
        return torch.where(mn < 1e-4, ix, torch.full_like(ix, -1)), mn

    player.restore(ck)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    import random
    random.seed(a.seed)
    obs = player.env_reset()
    player.get_batch_size(obs["obs"], 1)
    E = a.num_envs
    alive = torch.ones(E, dtype=torch.bool, device=task.device)
    prev_obj = task._target_states[:, 0:3].clone()
    prev_root = task._rigid_body_pos[:, 0, :].clone()
    rec = []
    torch.set_grad_enabled(False)
    for t in range(a.horizon):
        action = player.get_action(obs, False)
        obs_raw, _, done, inf = player.env_step(player.env, action)
        obs = player.obs_to_torch(obs_raw)
        alive = alive & ~inf["terminate"].bool()
        obj_now = task._target_states[:, 0:3].clone()
        root_now = task._rigid_body_pos[:, 0, :].clone()
        rin_obj = task._curr_obs[:, OBJ]
        rin_root = task._curr_obs[:, 0:3]
        label = (md.envid2sframe.long() + task.progress_buf.long())
        trace_f, trace_d = match(task._curr_ref_obs)
        rref = task._sync_ref_obs if man["state_sync"] else task._curr_ref_obs
        rew_f, rew_d = match(rref)
        phase_f = task.obs_buf[:, -1] * Lc
        hum = task._compute_humanoid_obs(None)
        objo = task._compute_obj_obs(None)
        ids = torch.arange(task.num_envs, device=task.device)
        row = task._hist_row_fixed(ids, hum, objo, False) if man["hist_layout_fix"] else None
        hist_err = (task._hist_obs_batch[:, -1] - row).abs().max(dim=-1).values if row is not None \
            else torch.full((E,), float("nan"), device=task.device)
        rec.append(dict(
            t=t, phys_step=t + 1, alive=alive.cpu().numpy(), progress=task.progress_buf.cpu().numpy(),
            label=label.cpu().numpy(), trace_ref=trace_f.cpu().numpy(), reward_ref=rew_f.cpu().numpy(),
            rin_obj_vs_now=(rin_obj - obj_now).norm(dim=-1).cpu().numpy(),
            rin_obj_vs_prev=(rin_obj - prev_obj).norm(dim=-1).cpu().numpy(),
            rin_root_vs_now=(rin_root - root_now).norm(dim=-1).cpu().numpy(),
            rin_root_vs_prev=(rin_root - prev_root).norm(dim=-1).cpu().numpy(),
            phase=phase_f.cpu().numpy(), hist_err=hist_err.cpu().numpy(),
            reward=task.rew_buf.cpu().numpy()))
        prev_obj, prev_root = obj_now, root_now

    def med(x):
        return float(np.median(x)) if len(x) else float("nan")
    print(f"\n[state_timing] policy {a.iteration} e{a.epoch} in env {env_it} ({man['task_class']}, stateSync="
          f"{man['state_sync']}), {E} envs, seed {a.seed}")
    print(" t | phys | label | trace ref | reward ref | phase*L | reward book vs s_t+1 / s_t (cm) | root (cm) | hist")
    steps = []
    for r in rec:
        m = r["alive"]
        if not m.any():
            break
        s = dict(t=r["t"], phys_step=r["phys_step"], n_alive=int(m.sum()),
                 label=med(r["label"][m]), trace_ref=med(r["trace_ref"][m]), reward_ref=med(r["reward_ref"][m]),
                 phase_frame=med(r["phase"][m]),
                 reward_book_vs_now_cm=100 * med(r["rin_obj_vs_now"][m]),
                 reward_book_vs_prev_cm=100 * med(r["rin_obj_vs_prev"][m]),
                 reward_book_vs_now_max_cm=100 * float(r["rin_obj_vs_now"][m].max()),
                 reward_root_vs_now_cm=100 * med(r["rin_root_vs_now"][m]),
                 reward_root_vs_prev_cm=100 * med(r["rin_root_vs_prev"][m]),
                 hist_row_err=float(np.nanmax(r["hist_err"][m])), reward=med(r["reward"][m]),
                 label_minus_trace=med((r["label"] - r["trace_ref"])[m]),
                 reward_minus_trace=med((r["reward_ref"] - r["trace_ref"])[m]))
        steps.append(s)
        if r["t"] < 4 or r["t"] in (50, 60, 70, 80, 100, 140):
            print(f"{s['t']:3d} | {s['phys_step']:4d} | {s['label']:5.0f} | {s['trace_ref']:9.0f} | "
                  f"{s['reward_ref']:10.0f} | {s['phase_frame']:7.2f} | {s['reward_book_vs_now_cm']:6.2f} / "
                  f"{s['reward_book_vs_prev_cm']:6.2f} | {s['reward_root_vs_now_cm']:5.2f} / "
                  f"{s['reward_root_vs_prev_cm']:5.2f} | {s['hist_row_err']:.1e}")
    body = [s for s in steps[1:] if s["trace_ref"] >= 0]    # t >= 1 (previous step exists), inside the clip
    w = [s for s in body if 45 <= s["t"] <= 85]
    summ = dict(
        label_minus_trace_ref=sorted({s["label_minus_trace"] for s in body}),
        reward_ref_minus_trace_ref=sorted({s["reward_minus_trace"] for s in body}),
        phase_minus_trace_ref=float(np.median([s["phase_frame"] - s["trace_ref"] for s in body])),
        reward_book_vs_now_cm_median=med([s["reward_book_vs_now_cm"] for s in body]),
        reward_book_vs_prev_cm_median=med([s["reward_book_vs_prev_cm"] for s in body]),
        reward_book_vs_now_cm_max_t45_85=max([s["reward_book_vs_now_max_cm"] for s in w], default=float("nan")),
        reward_book_vs_now_cm_median_t45_85=med([s["reward_book_vs_now_cm"] for s in w]),
        hist_row_err_max=max(s["hist_row_err"] for s in steps),
        step0=steps[0] if steps else None,
    )
    print(f"[state_timing] label - trace ref: {summ['label_minus_trace_ref']} | reward ref - trace ref: "
          f"{summ['reward_ref_minus_trace_ref']} | phase*L - trace ref (median): {summ['phase_minus_trace_ref']:.2f}")
    print(f"[state_timing] reward-input book vs this step's refreshed book: median {summ['reward_book_vs_now_cm_median']:.2f}"
          f" cm, t45-85 median {summ['reward_book_vs_now_cm_median_t45_85']:.2f} / max "
          f"{summ['reward_book_vs_now_cm_max_t45_85']:.2f} cm; vs previous step's: median "
          f"{summ['reward_book_vs_prev_cm_median']:.2f} cm | history row vs refreshed state max {summ['hist_row_err_max']:.1e}")
    print(f"[state_timing] effective: termination height {man['termination_height_effective']:.3f} m (env.yaml "
          f"{man['termination_height_configured']}), asset {man['asset_file']}, book mass {man['book']['mass']:.3f} kg, "
          f"book shapes {man['book']['n_shapes']} (friction {man['book']['shapes'][0]['friction']}, contact offset "
          f"{man['book']['shapes'][0]['contact_offset']:.3f}), sim substeps {man['sim']['substeps']} solver "
          f"{man['sim']['solver_type']} pos/vel iters {man['sim']['num_position_iterations']}/"
          f"{man['sim']['num_velocity_iterations']}, obs size {man['obs_size']}")
    out = a.out or os.path.join(L.paths(env_it)["iter_dir"], "probes", f"state_timing_{a.iteration}_e{a.epoch}")
    os.makedirs(out, exist_ok=True)
    L.write_json(os.path.join(out, "state_timing.json"),
                 dict(args=vars(a), checkpoint=L.rel(ck), manifest=man, summary=summ, steps=steps))
    print(f"[state_timing] -> {L.rel(os.path.join(out, 'state_timing.json'))}")


if __name__ == "__main__":
    main()
