#!/usr/bin/env bash
# SkillMimic-V2: Place Book (책상의 책을 집어 왼손으로 들고 옮기기, 150 frame), 고친 환경 (method fix, 프로젝트 루트 SKILLMIMIC_NOTES.md 4절)
#   run 폴더: experiments/runs/place_book_fix_n2048_s0  (업스트림 그대로 돌린 이전 run place_book_ours_n2048_s0은 비교용으로 남는다)
#   bash run_book.sh            # 학습 3000 epoch (약 9~10시간) -> 체크포인트 평가. 끊기면 같은 명령으로 이어서
#   EPOCHS=6000 bash run_book.sh # 예산 늘리기: 끝난 run도 3000부터 이어서 더 학습
# 자세한 설명: 프로젝트 루트 SKILLMIMIC.md, experiments/scripts/run_baseline.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/run_baseline.sh" place_book
