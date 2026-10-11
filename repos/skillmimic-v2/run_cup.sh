#!/usr/bin/env bash
# SkillMimic-V2: Drink Cup (식탁의 컵을 들어 마시기, 180 frame), 고친 환경 (method fix, 프로젝트 루트 SKILLMIMIC_NOTES.md 4절)
#   run 폴더: experiments/runs/drink_cup_fix_n2048_s0  (업스트림 그대로 돌린 이전 run drink_cup_ours_n2048_s0은 비교용으로 남는다)
#   bash run_cup.sh            # 학습 3000 epoch (약 10시간) -> 체크포인트 평가. 끊기면 같은 명령으로 이어서
#   EPOCHS=6000 bash run_cup.sh # 예산 늘리기: 끝난 run도 3000부터 이어서 더 학습
# 자세한 설명: 프로젝트 루트 SKILLMIMIC.md, experiments/scripts/run_baseline.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/run_baseline.sh" drink_cup
