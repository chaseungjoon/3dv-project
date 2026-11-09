# Local verification — 2026-11-09

Environment: macOS arm64, Python3.12.4 + numpy. Local probes found no torch, Isaac Gym, nvidia-smi or ShellCheck. No robot learning, CUDA evaluation, or real checkpoint restore was performed.

|Check|Status|Evidence/scope|
|---|---|---|
|Task Avoidance CPU tests|PASS|42 unittest cases (17 legacy +25 v2); synthetic/mocked only|
|Existing progress regression tests|PASS|13 unittest cases|
|Bash syntax|PASS|bash -n on all5 task_avoidance wrappers|
|Python syntax|PASS|AST parse of12 task_avoidance modules +shared runner|
|git whitespace|PASS|git diff --check|
|Frozen fix/code/config preservation|PASS|No diff against d8767fd in skillmimic/, configs/env/, configs/train/|
|Original experiment worktree preservation|PASS|Original /Desktop/3dv-experiments git status clean; separate clone used|
|ShellCheck|NOT RUN|Not installed|
|verify_gpu integration|NOT RUN|Requires team's prepared CUDA/Isaac Gym machine, data, encoder, policy|
|Actual checkpoint/frame/continuation validation|NOT RUN|CPU fixtures validate orchestration, not real checkpoint contents|
|JIT reward factors/delivered reward equivalence|NOT RUN|Runtime assertion implemented, not GPU-executed|
|Real initial pairing/repeatability/donor handoff|NOT RUN|Implemented, CPU synthetic control analysis passed; no real trajectory|
|Reach proxy video validation|NOT RUN|Pilot thresholds; recording/manual-label agreement tooling not implemented|
|Task Avoidance hypothesis|NOT RUN|No new empirical evaluation/training|

## Reproduce local checks

From repos/skillmimic-v2:

```bash
python3 -B -m unittest discover -s experiments/tools/task_avoidance -p 'test*.py'
python3 -B -m unittest discover -s experiments/tools/progress_reward -p 'test*.py'
```

New regressions cover second-worker launch failure, analyzer launch failure, interruption, strict JSON, exact frame accounting, wrong checkpoint epoch, target-already-saved resume auditing without training, failed completion audit, dry-run command construction/no writes, failed-lift reach vs partial lift, object approaching stationary wrist, post-terminal exclusion, switched-tail eligibility, nonfinite geometry/action/initial state, shape errors, duplicate initial states/episode IDs, result checkpoint identity, missing/explicit pairs, wrist repeat divergence, self-reload controls, and empty eligible populations.

The target-resume tests use mock checkpoint audit results. They prove the orchestrator does not launch training at the target; only the GPU-machine audit will inspect real torch checkpoint payloads. The synthetic donor-prefix tests exercise control gating and aggregation, not simulator continuity. Reward-source AST testing is not numerical torch equivalence.

## GPU checks to run next

Use README steps1–3 before learning. verify_gpu calls the actual fix_progress training task, compares the real parent reward to delivered reward with weight0/1, explicitly drives the buffer path, records terminal coverage, and checks partial reset isolation/history/object state and distance caching. No optimizer updates. Verification uses4 environments and is not a2048-environment scaling test.

Capability evaluation checks identical recorded switch states, restore invariance, donor-uninterrupted vs self-reload trajectories and candidate repeats. A failed control yields INCONCLUSIVE and suppresses paired conclusion data. Snapshot equality does not serialize hidden PhysX solver state; the trajectory control checks are essential.

After those diagnostics and prespecified video review, explicitly decide whether to run20epoch C0/C1 smoke, inspect checkpoint completion.json/frame deltas, and allocate the500epoch comparison. No GPU-learning approval is implied by a CPU PASS. Full physics validation, held-out confirmatory seeds, and automatic video/manual-label calibration remain outside this implementation.
