#!/usr/bin/env bash
# Phase 6: 실행 가능성(viability) screening. 가중치는 그대로, closed-loop 출력 변환만 바꿔서
# baseline과 "같은 episode"에서 성공률과 잡기 단계 비율이 움직이는지 본다 (paired McNemar, REPORT의 Phase 6 표).
#   sim_notask       언어를 빼도 같은가 (언어를 안 쓴다는 오프라인 결과의 closed-loop 확인)
#   sim_no_ensemble  chunk ensemble을 끄면
#   sim_scale1.5/2/3/5  action_scale (closed-loop 움직임이 데이터보다 2~5배 작았음)
# suite: WidowX 전체(96) + Google 원래 URDF subset (coke can 75, move near 60). variant당 약 24분, 기본 6개 약 2.5시간 (추정).
#   사용법: bash experiments/scripts/phase6_viability.sh
#           VARIANTS="sim_scale2 sim_scale3" bash experiments/scripts/phase6_viability.sh   # 일부만
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
t10_require_sim
VARIANTS="${VARIANTS:-sim_notask sim_no_ensemble sim_scale1.5 sim_scale2 sim_scale3 sim_scale5}"
SUITES="${SUITES:-bridge coke_can_quick move_near_quick}"
for v in $VARIANTS; do
  for s in $SUITES; do run_sim "$s" --variant "$v"; done
  bash experiments/scripts/report.sh >/dev/null   # variant마다 리포트 갱신 (중간에 멈춰도 결과를 볼 수 있게)
done
bash experiments/scripts/report.sh
