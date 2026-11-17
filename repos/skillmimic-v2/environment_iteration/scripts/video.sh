#!/usr/bin/env bash
# Render failure clips of one iteration (GPU, ~1-3 min per rollout). Agent may run it.
#
#   bash environment_iteration/scripts/video.sh fix-2                     # clips requested in iterations/fix-2/clips.json
#   bash environment_iteration/scripts/video.sh fix-2 --cell stoch --seed 101 --window 60:110   # ad hoc window, all envs
#   any src/render.py option passes through (--epoch, --num_envs, --horizon, --slowmo, --size, --tag, --clips)
# Output: environment_iteration/video/<it>/e<epoch>_<cell>_s<seed>[_tag]/INDEX.md (+ full/, clips/, sheets/)
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
smv2_require_setup
ei_iteration "${1:-}"; shift
ei_gpu_free_or_exit
"$PY" "$EI_SRC/render.py" "$ITERATION" "$@" 2>&1 | grep --line-buffered -E '^\[render|^\[rollout|Error|Traceback|error:|no clips|no checkpoint'
exit "${PIPESTATUS[0]}"
