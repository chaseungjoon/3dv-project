#!/usr/bin/env bash
# Phase 5: 시뮬레이터 검증. 같은 설치에서 SimplerEnv 공식 Octo wrapper + 공식 Octo-Base가 논문 값을 재현하는지 본다.
# 재현되면 "CrossFormer 성공률이 낮다"는 것이 우리 설치 문제가 아니라는 근거가 된다.
#   WidowX 4 task x 24 episode를 seed(init_rng) 0, 2, 4로 (논문과 같은 방식), Google은 원래 URDF subset을 seed 0으로.
# 약 45분 (추정).
#   사용법: bash experiments/scripts/phase5_sim_validity.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
t10_require_sim
t10_require_octo
for rng in 0 2 4; do run_sim bridge --policy octo-base --octo-init-rng "$rng"; done
for s in coke_can_quick move_near_quick; do run_sim "$s" --policy octo-base --octo-init-rng 0; done
bash experiments/scripts/report.sh
