#!/usr/bin/env bash
# 파이프라인 점검: 시뮬레이터, WBC, 학습, 평가, 리포트를 아주 짧게 한 번씩. 숫자 자체는 의미 없다.
# 결과는 experiments/runs/_check, experiments/results/_check 에만 쓴다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hmg_require_setup
export RUNS_DIR="$HMG_REPO/experiments/runs/_check" RESULTS_DIR="$HMG_REPO/experiments/results/_check"
rm -rf "$RUNS_DIR" "$RESULTS_DIR"; mkdir -p "$RUNS_DIR" "$RESULTS_DIR"
fail() { echo "파이프라인 점검 실패: $*" >&2; exit 1; }

hmg_log "1/5 무작위 행동 300 step ($TASK 환경 + WBC)"
"$PY" scripts/demo_random_action.py --task LMPushButton --steps 300 --seed 0 --video-path "$RESULTS_DIR/random_action.mp4" 2>&1 \
  | grep -E "saved video|Base displacement norm" || fail "random action"
hmg_log "2/5 사람 시범 1개를 WBC로 재생 (task 성공해야 함)"
"$PY" scripts/playback_dataset.py "experiments/data/source_demo_replay/datasets/$TASK/demo.hdf5" --action-source wbc-goal \
  --allow-state-divergence --require-task-success --video-path "$RESULTS_DIR/source_demo_wbc.mp4" 2>&1 \
  | grep -E "task success:|Playback completed" || fail "source demo playback"
hmg_log "3/5 학습 200 step (100마다 저장)"
SAVE_FREQ=100 bash experiments/scripts/train.sh 200 || fail train
hmg_log "4/5 평가 2 episode (step 200)"
bash experiments/scripts/eval.sh 2 000200 || fail eval
ls "$RESULTS_DIR"/eval/*/eval_000200_2ep.json >/dev/null || fail "평가 결과 없음"
hmg_log "4b/5 기준 정책 (mean, 2 episode)"
"$PY" experiments/tools/eval_baselines.py --policy mean --task "$TASK" --num-episodes 2 --data-root "$DATA_DIR/projected" \
  --output "$RESULTS_DIR/baselines/mean_2ep.json" 2>&1 | grep -E "success_rate" || fail baselines
hmg_log "5/5 리포트"
"$PY" experiments/tools/report.py --runs_dir "$RUNS_DIR" --results_dir "$RESULTS_DIR" --out "$RESULTS_DIR/report" || fail report
hmg_log "파이프라인 점검 통과 ($RESULTS_DIR/report/REPORT.md)"
