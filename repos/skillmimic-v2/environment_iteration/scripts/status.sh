#!/usr/bin/env bash
# Refresh the results table in STATUS.md and print it (CPU, instant).
#
#   bash environment_iteration/scripts/status.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$PY" "$EI_SRC/status.py" && "$PY" "$EI_SRC/status.py" --print
