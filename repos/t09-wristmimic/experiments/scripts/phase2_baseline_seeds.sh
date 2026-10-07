#!/usr/bin/env bash
# Phase 2: 기본 설정 seed 1, 2를 예산 T로 학습 (seed 0은 Phase 1 run의 epoch T 체크포인트 사용).
#   사용법: BUDGET=<T> bash experiments/scripts/phase2_baseline_seeds.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t09_require_budget
bash experiments/scripts/queue.sh "$BUDGET" default:1 default:2
