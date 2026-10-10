#!/usr/bin/env bash
# 세 baseline run(run_cup.sh, run_book.sh, run_kettle.sh)의 결과를 프로젝트 루트의 BASELINE_SKILLMIMIC.md로 정리한다.
#   bash write_results.sh
# GPU를 쓰지 않는다 (수 초). 학습 중에 실행해도 된다: 끝나지 않은 run은 상태와 이어서 할 명령을 적는다.
# 그림/표: experiments/results/figures/, experiments/results/tables/
source "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/_common.sh"
"$PY" experiments/tools/write_results.py "$@" 2>&1 | grep -v -i warn
