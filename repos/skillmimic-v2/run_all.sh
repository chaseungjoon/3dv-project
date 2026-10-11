#!/usr/bin/env bash
# SkillMimic-V2 baseline 2차: 컵 -> 책 -> 결과 문서까지 자동으로. 둘 다 고친 환경(method fix, 프로젝트 루트 SKILLMIMIC_NOTES.md 4절).
#   bash run_all.sh     # tmux 안에서 권장. 남은 컵(약 1시간) + 책(약 10시간) + 평가(clip당 약 15분)
#
# 순서 (clip마다 experiments/scripts/run_baseline.sh = 학습 -> 평가):
#   1. 컵 drink_cup_fix_n2048_s0: 마지막 체크포인트부터 이어서 3000 epoch -> 평가 (250 epoch마다 + 최종 3가지)
#   2. 책 place_book_fix_n2048_s0: 처음부터 3000 epoch -> 평가
#   3. bash write_results.sh -> 프로젝트 루트에 새 문서 BASELINE_SKILLMIMIC_2.md (기존 문서는 덮어쓰지 않음, 다음 번호)
# 끊겨도 같은 명령을 다시 실행하면 된다: 끝난 학습과 평가는 건너뛰고, 끊긴 곳(10 epoch마다 체크포인트)부터 이어서 한다.
# 학습이 죽으면 (예: GPU 메모리 부족) run_baseline.sh가 2번, 그래도 안 되면 여기서 RETRIES번(기본 2) 더 이어서 한다.
#   그래도 안 되는 clip은 건너뛰고 다음 clip으로 간다. 결과 문서는 항상 마지막에 쓴다 (끝나지 않은 clip은 상태로 적힌다).
#   Ctrl-C는 다시 시도하지 않고 바로 멈춘다 (결과 문서도 쓰지 않음).
# 다른 SkillMimic 학습/평가가 이미 돌고 있으면 (예: run_cup.sh) 그게 끝날 때까지 기다렸다가 시작한다.
# 전체 출력은 experiments/logs/run_all_<날짜시각>.log 에도 남는다.
# 환경변수: CLIPS (기본 "drink_cup place_book". 주전자는 보류), EPOCHS, SEED, NUM_ENVS, EVAL_EVERY (run_baseline.sh와 같음), RETRIES
source "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/_common.sh"
smv2_require_setup
CLIPS="${CLIPS:-drink_cup place_book}"
RETRIES="${RETRIES:-2}"
EPOCHS="${EPOCHS:-3000}"
SEED="${SEED:-0}"
export EPOCHS SEED

mkdir -p "$RUNS_DIR" experiments/logs
exec 9> "$RUNS_DIR/.run_all.lock"
flock -n 9 || { smv2_log "run_all.sh가 이미 돌고 있다. 끝나기를 기다리거나 그 프로세스를 멈춘 뒤 다시 실행"; exit 1; }
LOG="experiments/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# GPU를 같이 쓰면 둘 다 메모리 부족으로 죽는다 (12GB). 이 저장소의 다른 학습/평가가 끝날 때까지 기다린다.
others() { pgrep -f -- "$SMV2_REPO/.venv/bin/python|run_baseline.sh" | grep -v -x "$$"; }
if [[ -n "$(others)" ]]; then
  smv2_log "다른 SkillMimic 작업이 돌고 있다 (PID $(others | tr '\n' ' ')). 끝나면 시작한다 (1분마다 확인)"
  while [[ -n "$(others)" ]]; do sleep 60; done
  sleep 30   # GPU 메모리가 풀리도록
fi

status() {  # clip 하나의 진행 상태 한 줄
  local run="${1}_$(smv2_method "$1")_n${NUM_ENVS}_s${SEED}" ep n
  ep=$("$PY" experiments/tools/ckpt.py latest "$RUNS_DIR/$run" 2>/dev/null | cut -f2)
  n=$(find "$RESULTS_DIR/eval/det" "$RESULTS_DIR/eval/stoch" "$RESULTS_DIR/eval/stoch_perturb" -path "*/$run/*_$(printf %08d "$EPOCHS").json" 2>/dev/null | wc -l)
  if [[ -z "$ep" ]]; then echo "$run: 시작 안 함 -> 처음부터"
  elif (( ep < EPOCHS )); then echo "$run: 학습 중단됨 (epoch $ep/$EPOCHS) -> 이어서 학습"
  elif (( n < 3 )); then echo "$run: 학습 끝, 평가 남음"
  else echo "$run: 완료 -> 건너뜀"; fi
}
smv2_log "=== run_all 시작: $CLIPS, 목표 $EPOCHS epoch, seed $SEED, env $NUM_ENVS. 로그 $LOG ==="
for c in $CLIPS; do smv2_log "  $(status "$c")"; done

FAILED=""
for c in $CLIPS; do
  for try in $(seq 0 "$RETRIES"); do
    (( try > 0 )) && smv2_log "=== $c: 다시 시도 $try/$RETRIES (마지막 체크포인트부터) ==="
    bash experiments/scripts/run_baseline.sh "$c"
    code=$?
    (( code == 0 )) && break
    if (( code == 130 )); then smv2_log "중단 (Ctrl-C). 다시 실행하면 $c부터 이어서 한다."; exit 130; fi
    if (( try == RETRIES )); then
      smv2_log "$c: $((RETRIES + 1))번 시도했지만 끝나지 않았다 (exit $code). 다음 clip으로 넘어간다. 로그: $RUNS_DIR/${c}_$(smv2_method "$c")_n${NUM_ENVS}_s${SEED}/train.log"
      FAILED="$FAILED $c"
      break
    fi
    sleep 60
  done
done

smv2_log "=== 결과 문서 작성 (write_results.sh) ==="
bash write_results.sh
for c in $CLIPS; do smv2_log "  $(status "$c")"; done
if [[ -n "$FAILED" ]]; then
  smv2_log "=== run_all 끝, 끝나지 않은 clip:$FAILED. 원인을 고친 뒤 bash run_all.sh 를 다시 실행하면 이어서 한다 (결과 문서는 새 번호로 다시 쓴다) ==="
  exit 1
fi
smv2_log "=== run_all 끝: $CLIPS 학습, 평가, 결과 문서 완료. 로그 $LOG ==="
