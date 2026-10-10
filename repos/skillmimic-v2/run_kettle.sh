#!/usr/bin/env bash
# SkillMimic-V2 baseline: Place Kettle (주전자를 들어 옮겨 놓기, 100 frame. T09와 같은 물체)
#   bash run_kettle.sh            # 학습 3000 epoch (약 9~10시간) -> 체크포인트 평가. 끊기면 같은 명령으로 이어서
#   EPOCHS=6000 bash run_kettle.sh # 예산 늘리기: 끝난 run도 3000부터 이어서 더 학습
# 자세한 설명: QUICKSTART.md, experiments/scripts/run_baseline.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/run_baseline.sh" place_kettle
