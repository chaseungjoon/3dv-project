#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
# Orchestration needs only Python stdlib. PY selects the existing GPU Python.
exec "${ORCHESTRATOR_PY:-python3}" "$REPO/experiments/tools/progress_reward/runner.py" preflight "$@"
