#!/usr/bin/env bash
# 세 baseline run(run_cup.sh, run_book.sh, run_kettle.sh)의 결과를 프로젝트 루트에 새 문서로 정리한다.
#   bash write_results.sh
# 기존 문서는 덮어쓰지 않는다: BASELINE_SKILLMIMIC.md가 없으면 그 이름으로, 있으면 BASELINE_SKILLMIMIC_2.md, _3.md, ...
#   (가장 큰 번호 + 1). 그림도 문서마다 따로 (baseline_curves_N.png). 표(csv)는 최신 것만.
# GPU를 쓰지 않는다 (수 초). 학습 중에 실행해도 된다: 끝나지 않은 run은 상태와 이어서 할 명령을 적는다.
# 그림/표: experiments/results/figures/, experiments/results/tables/. 특정 파일에 쓰려면 --out <경로> (덮어씀)
source "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/_common.sh"
"$PY" experiments/tools/write_results.py "$@" 2>&1 | grep -v -i warn
