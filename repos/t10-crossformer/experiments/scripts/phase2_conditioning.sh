#!/usr/bin/env bash
# Phase 2: task 조건과 history 길이 (모두 같은 frozen 가중치, 추론 방식만 다름). GPU 약 20분 (추정).
#   cond_goal_nb    notebook의 goal 조건 (언어 자리에 USE("") 사용)
#   cond_goal_sub8  oracle sub-goal: 8 step 뒤 frame을 goal로 (학습의 goal relabeling 분포에 가까움, 배포 불가)
#   cond_none       task 없음 (언어도 goal도 0)
#   hist_w1/hist_w2 history 1, 2 frame (학습은 5)
#   사용법: bash experiments/scripts/phase2_conditioning.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
for v in cond_goal_nb cond_goal_sub8 cond_none hist_w1 hist_w2; do run_offline "$v"; done
bash experiments/scripts/report.sh
