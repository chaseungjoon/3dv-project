#!/usr/bin/env bash
# SkillMimic-V2 baseline 세 clip을 순서대로: 컵 -> 책 -> 주전자. clip마다 학습 3000 epoch -> 평가.
#   bash run_all.sh          # 약 28시간 + 평가 30분. tmux 안에서 권장
#
# 끊겨도 같은 명령을 다시 실행하면 된다. 컵부터 차례로 확인해서
#   학습과 평가가 다 끝난 clip은 건너뛰고, 학습이 끊긴 clip은 마지막 체크포인트(10 epoch마다)부터 이어서,
#   학습만 끝난 clip은 남은 평가만 하고 다음 clip으로 넘어간다.
# clip이 죽으면 (예: GPU 메모리 부족) 마지막 체크포인트부터 최대 RETRIES번 (기본 2) 다시 이어서 한다.
#   그래도 안 되면 멈춘다 (다시 실행하면 그 clip부터). Ctrl-C는 다시 시도하지 않고 바로 멈춘다.
# 다른 SkillMimic 학습/평가가 이미 돌고 있으면 (예: 손으로 실행한 run_cup.sh) 그게 끝날 때까지 기다렸다가 시작한다.
# 결과 문서는 만들지 않는다. 끝나면: bash write_results.sh
# 환경변수: EPOCHS, SEED, NUM_ENVS, EVAL_EVERY (run_baseline.sh와 같음), RETRIES
source "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/_common.sh"
smv2_require_setup
CLIPS="drink_cup place_book place_kettle"
RETRIES="${RETRIES:-2}"
EPOCHS="${EPOCHS:-3000}"
SEED="${SEED:-0}"
export EPOCHS SEED

mkdir -p "$RUNS_DIR"
exec 9> "$RUNS_DIR/.run_all.lock"
flock -n 9 || { smv2_log "run_all.sh가 이미 돌고 있다. 끝나기를 기다리거나 그 프로세스를 멈춘 뒤 다시 실행"; exit 1; }

# GPU를 같이 쓰면 둘 다 메모리 부족으로 죽는다 (12GB). 이 저장소의 다른 학습/평가가 끝날 때까지 기다린다.
others() { pgrep -f -- "$SMV2_REPO/.venv/bin/python|run_baseline.sh|run_after.sh" | grep -v -x "$$"; }
if [[ -n "$(others)" ]]; then
  smv2_log "다른 SkillMimic 작업이 돌고 있다 (PID $(others | tr '\n' ' ')). 끝나면 시작한다 (1분마다 확인)"
  while [[ -n "$(others)" ]]; do sleep 60; done
  sleep 30   # GPU 메모리가 풀리도록
fi

status() {  # clip 하나의 진행 상태 한 줄
  local run="${1}_ours_n${NUM_ENVS}_s${SEED}" ep n
  ep=$("$PY" experiments/tools/ckpt.py latest "$RUNS_DIR/$run" 2>/dev/null | cut -f2)
  n=$(find "$RESULTS_DIR/eval/det" "$RESULTS_DIR/eval/stoch" "$RESULTS_DIR/eval/stoch_perturb" -path "*/$run/*_$(printf %08d "$EPOCHS").json" 2>/dev/null | wc -l)
  if [[ -z "$ep" ]]; then echo "시작 안 함"
  elif (( ep < EPOCHS )); then echo "학습 중단됨 (epoch $ep/$EPOCHS) -> 이어서 학습"
  elif (( n < 3 )); then echo "학습 끝, 평가 남음"
  else echo "완료 -> 건너뜀"; fi
}
smv2_log "=== run_all 시작: 목표 $EPOCHS epoch, seed $SEED, env $NUM_ENVS ==="
for c in $CLIPS; do smv2_log "  $c: $(status "$c")"; done

for c in $CLIPS; do
  for try in $(seq 0 "$RETRIES"); do
    (( try > 0 )) && smv2_log "=== $c: 다시 시도 $try/$RETRIES (마지막 체크포인트부터) ==="
    bash experiments/scripts/run_baseline.sh "$c"
    code=$?
    (( code == 0 )) && break
    if (( code == 130 )); then smv2_log "중단 (Ctrl-C). 다시 실행하면 $c부터 이어서 한다."; exit 130; fi
    if (( try == RETRIES )); then
      smv2_log "$c: $((RETRIES + 1))번 시도했지만 끝나지 않았다 (exit $code). 로그: $RUNS_DIR/${c}_ours_n${NUM_ENVS}_s${SEED}/train.log"
      smv2_log "원인을 고친 뒤 bash run_all.sh 를 다시 실행하면 $c부터 이어서 한다."
      exit "$code"
    fi
    sleep 60
  done
done
smv2_log "=== run_all 끝: 세 clip 학습과 평가 완료. 결과 정리: bash write_results.sh ==="
