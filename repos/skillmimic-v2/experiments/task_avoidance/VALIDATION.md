# Local verification — 2026-11-07

Environment: macOS arm64, Python3 + numpy. No CUDA/Isaac Gym/policy checkpoints locally.

|Check|Status|Evidence/scope|
|---|---|---|
|New CPU tests|PASS|17 unittest cases|
|Existing progress regression tests|PASS|13 unittest cases|
|Bash syntax|PASS|bash -n on all3 new wrappers|
|Python syntax|PASS|AST parse of all new Python files|
|git whitespace check|PASS|git diff --check|
|ShellCheck|NOT RUN|not installed|
|GPU integration|NOT RUN|requires existing team machine|
|Actual checkpoint restore/continuation|NOT RUN|no local policy checkpoint or GPU|
|Reward diagnostic equals actual JIT/delivered reward|NOT RUN|runtime assertion implemented, not executed locally|
|Initial state pairing and repeatability|NOT RUN|runtime snapshots/assertions implemented, not GPU-validated|
|Task Avoidance hypothesis|NOT RUN|no new robot evaluation/training performed|

CPU tests cover terminal inclusion/post-terminal exclusion, joint success within one episode, initial-position goal progress, history mismatch/nonfinite rejection, env-ID pairing rather than list ordering, reward-source AST structure, repeat divergence→INCONCLUSIVE, invalid protocol, missing epochs, paths with spaces, branch dry-run/no writes, explicit execute guard, fresh-process/reference command construction, failure stops later jobs, resampled reference window, invalid label, end-to-end synthetic analysis and output preservation. Synthetic data are not empirical robot results. AST structure testing is not numeric torch equivalence testing.

Source checks:
- `skillmimic/env/tasks/humanoid_task.py::post_physics_step`: reset checked before progress increment.
- `skillmimic/env/tasks/skillmimic_parahome_localhist.py::_init_with_random_noise`: initial reference phase can be resampled.
- `SkillMimicParahomeLocalHistRISBuffernodePhase._compute_reset`: object termination disabled in test mode.
- `skillmimic/env/tasks/skillmimic_parahome.py::compute_humanoid_reward`: multiplicative rb*ro*rig.
- `skillmimic/learning/common_agent.py::train`: stopping condition epoch > max_epochs; new branch launcher compensates without editing it.
- `skillmimic/learning/skillmimic_agent.py::restore_full_state`: existing model/optimizer/normalizer/ATS/RNG restore reused.
- `skillmimic/env/tasks/skillmimic_parahome.py::get_train_state`: serializes ATS, not full simulator physical/contact state.

Read-only inspection and successful CPU checks do not guarantee Isaac Gym runtime compatibility. Start with the explicit20epoch branch smoke and evaluate a small prespecified diagnostic subset before allocating the500epoch comparison. Preserve failures and protocol versions; do not loosen thresholds silently.
