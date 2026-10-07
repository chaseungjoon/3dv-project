#!/usr/bin/env bash
# Phase 4: SimplerEnv closed-loop 성공률 (visual matching, 공식 episode grid).
#   기본 suite: bridge(WidowX 4 task x 24 = 96 episode), coke_can(300), move_near(240) -> 약 1.5시간 (추정)
#   drawer(216 episode, ray tracing)는 느려서 기본에서 뺐다: SUITES="drawer" 로 따로 (약 1시간+, 추정)
#   variant 기본은 sim_baseline. 언어 없이 비교: VARIANTS="sim_notask"
#   사용법: bash experiments/scripts/phase4_closed_loop.sh
#           SUITES="bridge" VARIANTS="sim_baseline sim_notask" bash experiments/scripts/phase4_closed_loop.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
t10_require_sim
SUITES="${SUITES:-bridge coke_can move_near}"
VARIANTS="${VARIANTS:-sim_baseline}"
for v in $VARIANTS; do
  for s in $SUITES; do
    t10_log "closed-loop: $v / $s"
    "$PY" experiments/tools/sim_eval.py --suite "$s" --variant "$v" 2>&1 | t10_filter | grep -E '^\[sim|^->|rror'
  done
done
bash experiments/scripts/report.sh
