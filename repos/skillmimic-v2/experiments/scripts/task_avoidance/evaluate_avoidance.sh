#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
exec "${ORCHESTRATOR_PY:-python3}" "$REPO/experiments/tools/task_avoidance/runner.py" evaluate "$@"
