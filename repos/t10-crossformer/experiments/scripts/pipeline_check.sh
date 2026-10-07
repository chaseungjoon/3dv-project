#!/usr/bin/env bash
# 파이프라인 점검 (약 10분). 모든 단계를 아주 작게 한 번씩 돌려 스크립트가 끝까지 도는지 확인한다.
# 결과는 experiments/runs/_check, experiments/results/_check 에만 쓴다 (본 결과와 섞이지 않음).
#   사용법: bash experiments/scripts/pipeline_check.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
t10_require_sim

export T10_RUNS="$T10_REPO/experiments/runs/_check"
export T10_RESULTS="$T10_REPO/experiments/results/_check"
rm -rf "$T10_RUNS" "$T10_RESULTS"
N=4   # embodiment당 episode 수
fail=0

t10_log "1/5 action 규약 (episode $N개)"
"$PY" experiments/tools/conventions.py $DATASETS --max-episodes 20 2>&1 | t10_filter | tail -n 3 || fail=1

t10_log "2/5 오프라인 추론: Phase 1~3 variant 각 1개"
for v in baseline_goal baseline_lang cond_none rate_x2 obs_center_crop; do
  run_offline "$v" --max-episodes "$N" | grep '^\[' || fail=1
done

t10_log "3/5 지표 + 단위 probe"
"$PY" experiments/tools/metrics.py 2>&1 | t10_filter | grep -E '^(baseline|cond|rate|obs)' || fail=1

t10_log "4/5 closed-loop: suite마다 job 1개 x episode 1개"
for s in bridge coke_can move_near drawer; do
  "$PY" experiments/tools/sim_eval.py --suite "$s" --variant sim_baseline --max-jobs 1 --max-episodes-per-job 1 2>&1 \
    | t10_filter | grep -E '^\[sim|^->' || fail=1
done

t10_log "4b/5 시뮬레이터 검증(Octo-Base)과 screening suite: episode 1개씩"
if [[ -f experiments/third_party/octo/octo/model/octo_model.py ]]; then
  run_sim bridge --policy octo-base --max-jobs 1 --max-episodes-per-job 1 || fail=1
else
  echo "Octo 없음 (setup.sh 필요)"; fail=1
fi
run_sim coke_can_quick --variant sim_scale2 --max-jobs 1 --max-episodes-per-job 1 || fail=1

t10_log "5/5 리포트"
"$PY" experiments/tools/aggregate.py 2>&1 | t10_filter | tail -n 1 || fail=1

# 결과 파일이 다 생겼는지 확인
for v in baseline_goal baseline_lang cond_none rate_x2 obs_center_crop; do
  for d in $DATASETS; do
    [[ -f "$T10_RESULTS/metrics/$v/${d}__self.json" ]] || { echo "없음: metrics/$v/${d}__self.json"; fail=1; }
  done
done
for s in bridge coke_can move_near drawer; do
  [[ -f "$T10_RESULTS/sim/sim_baseline/$s.json" ]] || { echo "없음: sim/sim_baseline/$s.json"; fail=1; }
done
[[ -f "$T10_RESULTS/sim/octo-base_rng0/bridge.json" ]] || { echo "없음: sim/octo-base_rng0/bridge.json"; fail=1; }
[[ -f "$T10_RESULTS/sim/sim_scale2/coke_can_quick.json" ]] || { echo "없음: sim/sim_scale2/coke_can_quick.json"; fail=1; }
[[ -f "$T10_RESULTS/report/REPORT.md" ]] || { echo "없음: report/REPORT.md"; fail=1; }

if (( fail == 0 )); then
  t10_log "파이프라인 점검 통과. 결과: $T10_RESULTS/report/REPORT.md (episode가 적어 숫자 자체는 의미 없음)"
else
  t10_log "파이프라인 점검 실패. 위 출력을 확인하세요 (T10_VERBOSE=1 로 전체 로그)."; exit 1
fi
