#!/usr/bin/env bash
# Failure analysis from stored evaluations (CPU, seconds) + STATUS.md refresh. Use after changing src/metrics.py.
#
#   bash environment_iteration/scripts/analyze.sh fix-2 [--epoch E]
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ei_iteration "${1:-}"; shift
"$PY" "$EI_SRC/analyze.py" "$ITERATION" "$@" && "$PY" "$EI_SRC/status.py"
