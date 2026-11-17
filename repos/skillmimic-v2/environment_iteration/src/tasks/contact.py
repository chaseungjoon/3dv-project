"""STATUS (2026-11-17): not used by any iteration (fix-2 investigation: object friction 2 added only ~5/56 holds;
the thick-finger + no-self-collision combination it was written for failed). Kept: it is the working pattern for
setting shape properties at actor creation.

fix-2: contact properties of the manipulated object and the hands, set from the env config at actor creation.

SkillMimicParahomeIterContact = fix-1's SkillMimicParahomeLocalHistRISBuffernodePhase + three optional env.yaml keys:

  objectShapeProps: {friction: 2.0}            # rigid shape properties of the manipulated object
  handShapeProps:   {friction: 1.5}            # ... of every hand body (wrists and finger segments)
  handSelfCollision: false                     # no collisions between shapes of the SAME hand (default true)

Allowed property names: friction, torsion_friction, rolling_friction, restitution. Without the keys the class
behaves exactly like its parent.
handSelfCollision: false puts every shape of the left hand in collision-filter bit 1 and of the right hand in bit 2
(shapes sharing a bit never collide), so fingers / palm of one hand do not collide with each other, while each hand
still collides with the object, the scene, the body and the other hand. Needed with thickened finger capsules: at
11 mm neighbouring fingers overlap and self-collision pushed them apart with ~300 N in free air.
Properties are applied while the actors are created (in _build_target /
_build_env), because shape properties changed after prepare_sim are not reliably applied on the GPU pipeline
(environment_iteration/src/probes/grip.py docstring).
"""
from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRISBuffernodePhase

ALLOWED = ("friction", "torsion_friction", "rolling_friction", "restitution")
HAND_BODY_KEYS = ("Wrist", "CMC", "MCP", "PIP", "DIP", "IP")


def _check(d, key):
    bad = [k for k in d if k not in ALLOWED]
    if bad:
        raise ValueError(f"env cfg {key}: unknown shape properties {bad} (allowed: {ALLOWED})")
    return {k: float(v) for k, v in d.items()}


class SkillMimicParahomeIterContact(SkillMimicParahomeLocalHistRISBuffernodePhase):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._obj_shape_props = _check(cfg["env"].get("objectShapeProps") or {}, "objectShapeProps")
        self._hand_shape_props = _check(cfg["env"].get("handShapeProps") or {}, "handShapeProps")
        self._hand_self_collision = bool(cfg["env"].get("handSelfCollision", True))
        self._contact_logged = False
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

    def _apply(self, env_ptr, handle, props_cfg, body_filter=None, tag=""):
        g = self.gym
        props = g.get_actor_rigid_shape_properties(env_ptr, handle)
        if body_filter is None:
            idx = range(len(props))
        else:
            names = g.get_actor_rigid_body_names(env_ptr, handle)
            ranges = g.get_actor_rigid_body_shape_indices(env_ptr, handle)
            idx = [s for b, nm in enumerate(names) if body_filter(nm)
                   for s in range(ranges[b].start, ranges[b].start + ranges[b].count)]
        before = {k: getattr(props[idx[0]], k) for k in props_cfg} if len(idx) else {}
        for s in idx:
            for k, v in props_cfg.items():
                setattr(props[s], k, v)
        g.set_actor_rigid_shape_properties(env_ptr, handle, props)
        if not self._contact_logged and len(idx):
            print(f"[contact] {tag}: {len(idx)} shape(s) {before} -> {props_cfg}", flush=True)

    def _hand_filters(self, env_ptr, handle):
        g = self.gym
        names = g.get_actor_rigid_body_names(env_ptr, handle)
        ranges = g.get_actor_rigid_body_shape_indices(env_ptr, handle)
        props = g.get_actor_rigid_shape_properties(env_ptr, handle)
        n = {1: 0, 2: 0}
        for b, nm in enumerate(names):
            if not any(k in nm for k in HAND_BODY_KEYS):
                continue
            bit = 1 if nm.startswith("jLeft") else 2 if nm.startswith("jRight") else 0
            for s in range(ranges[b].start, ranges[b].start + ranges[b].count):
                props[s].filter = props[s].filter | bit
                n[bit] = n.get(bit, 0) + 1
        g.set_actor_rigid_shape_properties(env_ptr, handle, props)
        if not self._contact_logged:
            print(f"[contact] hand self-collision off: left {n[1]} shapes (filter bit 1), right {n[2]} (bit 2)",
                  flush=True)

    def _build_target(self, env_id, env_ptr):
        super()._build_target(env_id, env_ptr)
        if self._obj_shape_props:
            self._apply(env_ptr, self.target_handle, self._obj_shape_props, tag="object")

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self._hand_shape_props:
            self._apply(env_ptr, self.humanoid_handles[-1], self._hand_shape_props,
                        body_filter=lambda nm: any(k in nm for k in HAND_BODY_KEYS), tag="hand")
        if not self._hand_self_collision:
            self._hand_filters(env_ptr, self.humanoid_handles[-1])
        if env_id == 0:
            self._contact_logged = True
