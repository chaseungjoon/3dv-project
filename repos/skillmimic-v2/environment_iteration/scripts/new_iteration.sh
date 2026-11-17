#!/usr/bin/env bash
# Create the next iteration folder from its parent (copies iteration.json + env.yaml, writes templates).
#
#   bash environment_iteration/scripts/new_iteration.sh fix-2 "one-line summary of the change" [--from fix-1]
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -ge 2 ]] || { sed -n '2,4p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
NAME="$1" SUMMARY="$2"; shift 2
"$PY" "$EI_SRC/new_iteration.py" "$NAME" --summary "$SUMMARY" "$@" && "$PY" "$EI_SRC/status.py"
