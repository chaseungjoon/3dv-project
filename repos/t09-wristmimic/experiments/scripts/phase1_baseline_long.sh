#!/usr/bin/env bash
# Phase 1: 기본 설정 seed 0을 길게 1회 학습 -> 학습 곡선으로 학습 예산 T를 정한다.
#   사용법: bash experiments/scripts/phase1_baseline_long.sh
#   EPOCHS_LONG (기본 6000, 500의 배수). 예상 시간은 README 4절 표 (실측 epoch 시간 x EPOCHS_LONG).
# 이 run의 체크포인트(500, 1000, ...)는 이후 default seed 0의 예산 T 결과로 그대로 재사용한다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash experiments/scripts/queue.sh "${EPOCHS_LONG:-6000}" default:0
