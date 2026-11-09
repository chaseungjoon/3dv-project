# Local validation — 2026-11-07

- PASS: 13 CPU unit tests (`python3 -m unittest discover -s experiments/tools/progress_reward -p test_cpu.py`). Checks signed/clipped math, zero-weight arithmetic, invalid input, sequential A/B order and failure propagation, actual fake child exit7/log, explicit-main guard, dry-run no writes, existing output/resume fingerprint guard, archive checkpoint exclusion and overwrite refusal.
- PASS: bash -n for all five wrappers; Python AST parsing; git diff --check.
- PASS: main A/B dry-run emits separate commands with original fix configuration and explicit new task; does not launch GPU.
- PASS: existing JSON/NPZ analysis processed 65 evaluations to summary.csv and curves.svg. These are historical results, not new training.
- PASS: local preflight reports missing Linux/GPU/Python/encoder/scene and exits2 without installation.
- NOT RUN: ShellCheck (not installed).
- NOT RUN: Isaac Gym import, GPU training, actual task same-state weight0 equivalence, reset/buffer/terminal simulation, Ctrl+C during GPU execution, full-state resume fidelity, inference, GPU memory peak, policy evaluation and viewer/video recording.
- INCONCLUSIVE: book processed endpoint's semantic meaning; runtime goal logs and reference playback required. No claim of release/placement.

CPU math tests are not simulator integration tests. Historical analysis plots and CSV can be regenerated using TEAM_RUN_GUIDE. A successful wrapper/preflight is not proof that learning or the physical environment is correct. GPU integration must pass EXP02 before committing EXP03 compute.
