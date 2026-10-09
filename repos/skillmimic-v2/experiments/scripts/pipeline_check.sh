#!/usr/bin/env bash
# 파이프라인 점검 (약 15분). 모든 단계를 아주 짧게 한 번씩 돌린다. 숫자 자체는 의미 없다.
# 결과는 experiments/runs/_check, experiments/results/_check 에만 쓴다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
smv2_require_setup
export RUNS_DIR=experiments/runs/_check
export SMV2_TRAIN_CFG=experiments/configs/train/parahome_check.yaml   # 10 epoch마다 저장
rm -rf "$RUNS_DIR" experiments/results/_check
fail() { echo "파이프라인 점검 실패: $*" >&2; exit 1; }

smv2_log "1/5 ours 학습 20 epoch (2048 env)"
bash experiments/scripts/train.sh drink_cup ours 20 || fail "ours 학습"
smv2_log "2/5 같은 run 이어서 30 epoch까지 (자동 resume)"
bash experiments/scripts/train.sh drink_cup ours 30 || fail "resume"
grep -q "resume_from" "$RUNS_DIR/drink_cup_ours_n${NUM_ENVS}_s0/run_meta.json" || fail "resume 기록 없음"
smv2_log "3/5 sm 학습 20 epoch"
bash experiments/scripts/train.sh drink_cup sm 20 || fail "sm 학습"

smv2_log "4/5 평가 (det 곡선, final det/stoch/perturb)"
for r in drink_cup_ours_n${NUM_ENVS}_s0 drink_cup_sm_n${NUM_ENVS}_s0; do
  "$PY" experiments/tools/evaluate.py --run_dir "$RUNS_DIR/$r" --which all --mode det --out_dir experiments/results/_check/eval 2>&1 | grep -E '^\[eval|Error|Traceback' || fail "평가 $r"
done
"$PY" experiments/tools/evaluate.py --run_dir "$RUNS_DIR/drink_cup_ours_n${NUM_ENVS}_s0" --which final --mode stoch --out_dir experiments/results/_check/eval 2>&1 | grep -E '^\[eval|Error|Traceback'
"$PY" experiments/tools/evaluate.py --run_dir "$RUNS_DIR/drink_cup_ours_n${NUM_ENVS}_s0" --which final --mode stoch --perturb --out_dir experiments/results/_check/eval 2>&1 | grep -E '^\[eval|Error|Traceback'
n=$(find experiments/results/_check/eval -name '*.json' | wc -l)
(( n >= 7 )) || fail "평가 결과 $n개 (7개여야 함: ours det 3 + sm det 2 + stoch 1 + perturb 1)"

smv2_log "5/5 리포트"
"$PY" experiments/tools/report.py --runs_dir "$RUNS_DIR" --eval_dir experiments/results/_check/eval --out experiments/results/_check/report | grep report || fail "report"
smv2_log "파이프라인 점검 통과  (experiments/results/_check/report/REPORT.md)"
