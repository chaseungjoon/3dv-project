"""Task classes that iterations add (fix-2, fix-3, ...). Upstream files under skillmimic/ are never edited for an
iteration; a new behaviour is a subclass here, selected by the iteration's iteration.json "task" field.

How to add one (AGENTS.md 5):
  1. write the class in a new module of this package, e.g. tasks/fix2.py:
         from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRISBuffernodePhase as Base
         class SkillMimicParahomeIterFix2(Base):
             ...
     Prefer cfg-driven switches (a new env.yaml key that defaults to the parent's behaviour) over hard-coded changes.
  2. add it to REGISTRY below:  "SkillMimicParahomeIterFix2": "fix2"
  3. set "task": "SkillMimicParahomeIterFix2" in iterations/fix-2/iteration.json
A class registered here is part of every later iteration's fingerprint (src/fingerprint.py), so never change the
behaviour of a class an earlier iteration uses: copy it into a new class instead.

register() is called by src/launch.py after isaacgym is imported and skillmimic/ is on sys.path.
"""
import importlib

# task class name -> module in this package
REGISTRY = {
    "SkillMimicParahomeIterContact": "contact",      # (unused) fix-2 investigation: objectShapeProps / handShapeProps env keys
    "SkillMimicParahomeIterContactOffset": "contact_offset",   # fix-3: handContactOffset / objectContactOffset
    "SkillMimicParahomeIterATS": "ats",                        # fix-4: atsSampler (ATS reaches the start-frame sampler)
    "SkillMimicParahomeIterOwnStart": "own_start",             # fix-5: ownStart* (starts from the policy's own post-pickup states)
    "SkillMimicParahomeIterObjRot": "obj_rot",                 # fix-7: objRotWeight (sign-invariant book rotation)
    "SkillMimicParahomeIterStateSync": "state_sync",           # fix-8: stateSync (reward / termination / capture on the post-step state)
}


def register(task_name):
    """Make task_name resolvable by skillmimic/utils/parse_task.py (it looks tasks up by name in its own globals)."""
    import utils.parse_task as PT
    if hasattr(PT, task_name):
        return "upstream"
    if task_name not in REGISTRY:
        raise SystemExit(f"task {task_name!r} is neither an upstream task (skillmimic/utils/parse_task.py) nor in "
                         f"environment_iteration/src/tasks/__init__.py REGISTRY")
    mod = importlib.import_module(f"{__name__}.{REGISTRY[task_name]}")
    setattr(PT, task_name, getattr(mod, task_name))
    return f"tasks.{REGISTRY[task_name]}"
