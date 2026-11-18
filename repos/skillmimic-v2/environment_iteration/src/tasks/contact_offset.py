"""fix-3: contact offset per shape instead of globally, set from the env config at actor creation.

SkillMimicParahomeIterContactOffset = fix-1/fix-2's SkillMimicParahomeLocalHistRISBuffernodePhase + two optional
env.yaml keys (env section):

  handContactOffset: 0.02      # PhysX contact offset (m) of every hand shape (wrists and finger segments)
  objectContactOffset: 0.02    # ... of every shape of the manipulated object

Without the keys the class behaves exactly like its parent (every shape keeps sim.physx.contact_offset).

Why: fix-2 (global contact_offset 0.005) never learned a grasp, while fix-1's policy keeps its grasp in fix-2's
solver when the offset is 0.02 again (fix-2 ITERATION.md section 8). Per shape keeps every non-hand contact (feet /
ground, body, desk) as in fix-2. Probes (fix-3 ITERATION.md section 1): the HAND shapes' offset is what matters
(hand 0.02 alone transfers fix-1's grasp; book 0.035 with hand 0.005 does not, so it is not the hand-book pair
distance). Cost: 20.2 s/epoch vs 21.4 s for the global 0.02 (so most of the extra cost over fix-2 comes from the hand shapes).
Properties are applied while the actors are created (_build_env / _build_target): shape properties changed after
prepare_sim are not reliably applied on the GPU pipeline (src/probes/grip.py docstring).
"""
from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRISBuffernodePhase

HAND_BODY_KEYS = ("Wrist", "CMC", "MCP", "PIP", "DIP", "IP")


class SkillMimicParahomeIterContactOffset(SkillMimicParahomeLocalHistRISBuffernodePhase):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        h, o = cfg["env"].get("handContactOffset"), cfg["env"].get("objectContactOffset")
        self._hand_co = None if h is None else float(h)
        self._obj_co = None if o is None else float(o)
        self._co_logged = set()
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

    def _set_offset(self, env_ptr, handle, value, body_filter=None, tag=""):
        g = self.gym
        props = g.get_actor_rigid_shape_properties(env_ptr, handle)
        if body_filter is None:
            idx = list(range(len(props)))
        else:
            names = g.get_actor_rigid_body_names(env_ptr, handle)
            ranges = g.get_actor_rigid_body_shape_indices(env_ptr, handle)
            idx = [s for b, nm in enumerate(names) if body_filter(nm)
                   for s in range(ranges[b].start, ranges[b].start + ranges[b].count)]
        if not idx:
            raise RuntimeError(f"contact offset {tag}: no shapes matched")
        before = props[idx[0]].contact_offset
        for s in idx:
            props[s].contact_offset = value
        g.set_actor_rigid_shape_properties(env_ptr, handle, props)
        if tag not in self._co_logged:
            print(f"[contact_offset] {tag}: {len(idx)} shape(s) contact_offset {before:.4f} -> {value:.4f}", flush=True)
            self._co_logged.add(tag)

    def _build_target(self, env_id, env_ptr):
        super()._build_target(env_id, env_ptr)
        if self._obj_co is not None:
            self._set_offset(env_ptr, self.target_handle, self._obj_co, tag="object")

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self._hand_co is not None:
            self._set_offset(env_ptr, self.humanoid_handles[-1], self._hand_co,
                             body_filter=lambda nm: any(k in nm for k in HAND_BODY_KEYS), tag="hand")
