#!/usr/bin/env bash
# Phase 3: 규약 불일치 probe. "변환을 틀리면 얼마나 나빠지는가"를 재서 T10 연구 질문의 크기를 정한다.
#   관측 쪽: obs_center_crop (비정사각 카메라를 crop), obs_hflip (카메라 좌우 반전)
#   제어 주기: rate_x2, rate_x3 (k frame마다 1번 관측, GT는 k step 합)
#   action 단위: 추가 추론 없음. metrics.py가 baseline 예측을 다른 통계로 unnormalize해서 계산 (리포트 표)
# GPU 약 15분 (추정).
#   사용법: bash experiments/scripts/phase3_mismatch.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
for v in obs_center_crop obs_hflip rate_x2 rate_x3; do run_offline "$v"; done
bash experiments/scripts/report.sh
