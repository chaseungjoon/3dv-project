# Task Avoidance 진단 실험 — 실행 안내

목적: 동일 fix에서 **조작 감소 → 기존 보상의 유인 → 보상 개입 반응**을 구별한다. 성공률 저하만으로 회피를 확정하지 않는다. 아래 프로그램은 준비된 실행 도구이며 GPU 실험 결과가 아니다.

## 파일과 단계

|단계|실행 파일|변수|통제|산출물|
|---|---|---|---|---|
|A 행동 비교 + B 보상 진단|scripts/task_avoidance/evaluate_avoidance.sh|정책 checkpoint|초기 상태/history/위상, fix, action 결정성, 평가 규칙|초기 snapshot, trajectory NPZ, 보상 성분, episode 지표|
|A/B 결과 집계|scripts/task_avoidance/analyze_avoidance.sh|비교할 정책 label|같은 seed와 env ID끼리 paired 비교|차이 CSV, 보상/조작 곡선 SVG, 검증 보고서|
|C 보상 개입|scripts/task_avoidance/run_branches.sh|기존 보상 vs Progress|동일 source checkpoint·복원 상태·추가 samples·paired RNG seed|새 run/checkpoint/보상 로그/분기 manifest|

경로는 모두 `repos/skillmimic-v2/experiments/` 아래다. 기존 fix task, 보상, 종료 구현은 수정하지 않는다. 새 진단 evaluator가 원 fix 경로를 호출한다. 이전 progress 패키지의 preflight/로그/프로세스/lock/결과 압축을 재사용한다.

## 0. 팀원 머신 준비

```bash
cd /path/to/3dv-project/repos/skillmimic-v2
export PY=/absolute/path/to/existing/isaacgym/python
# ORCHESTRATOR_PY는 numpy가 있는 CPU Python (기본 python3)
bash experiments/scripts/progress_reward/preflight.sh
```

기존 Linux/RTX 5070/Isaac Gym 환경과 book data, assets, encoder, 정책 checkpoint를 사용한다. 설치/다운로드는 자동으로 하지 않는다. numpy는 CPU 집계/runner에도 필요하다. 검증된 기존 환경을 임의 업그레이드하지 않는다.

## 1. A/B — 기존 7개 checkpoint 평가

```bash
OLD_RUN=experiments/runs/place_book_fix_n2048_s0
bash experiments/scripts/task_avoidance/evaluate_avoidance.sh \
  --run "$OLD_RUN" --label oldfix_s0 --dry-run
bash experiments/scripts/task_avoidance/evaluate_avoidance.sh \
  --run "$OLD_RUN" --label oldfix_s0
```

요구 epoch: 750/1000/1500/2000/2500/2750/3000. 누락 시 중단한다. 일부만 남았다면 전체 결과를 본 뒤 유리한 시점을 고르지 말고 protocol 복사본에 제외 사유와 후보를 먼저 기록한다. 기본 3 evaluation seeds ×32환경 ×7정책 =672 episodes + 첫 정책·첫 seed 반복32 episodes. 각 checkpoint/seed를 **새 프로세스**로 평가한다.

출력: `experiments/runs/task_avoidance/eval_oldfix_s0/`. 같은 label 재사용/덮어쓰기 금지. 실패 후 기존 디렉터리를 삭제하지 않고 새 label로 실행한다. GPU 작업은 순차 실행한다. checkpoint 읽기만 하며 기존 학습 결과를 바꾸지 않는다.

자동으로 CPU 분석까지 수행한다. `analysis/episodes.csv`, `per_eval_seed.csv`, `paired_episode_differences.csv`, `paired_seed_means.csv`, `curves.svg`, `report.json` 확인. 기본 primary pair는 1000→3000이며 모든 후보 곡선을 함께 보고한다. 이 pair는 이미 본 예비 결과에서 선택했으므로 탐색적 결과이다.

첫 정책/첫 seed 반복이 허용 오차를 넘으면 상태 INCONCLUSIVE, paired 결론용 CSV 생성을 막는다. 이는 GPU 비결정성을 없애는 장치가 아니라 확인하는 장치다. 나머지 정책까지 반복 검증됐다고 주장하지 않는다.

## 2. C — 동일 checkpoint에서 보상만 바꾸어 분기

source는 기존 `method=fix`, `clip=place_book`, `num_envs=2048` run의 numbered checkpoint여야 한다. `--source`는 실제 파일명을 확인해서 지정한다. 1000epoch는 예시이며 모든 A/B 평가 결과를 보고 유리한 source를 고르면 탐색적 선택임을 기록한다.

```bash
SOURCE=/absolute/path/to/place_book_fix_n2048_s0/nn/ACTUAL_NAME_00001000.pth
# 먼저 20epoch만 검증: source1000 →1020, C0/C1 각 1,310,720 추가 samples
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label pilot --stage smoke --extra-epochs 20 --seeds 0 --dry-run
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label pilot --stage smoke --extra-epochs 20 --seeds 0 --execute

# GPU smoke 통과 후. 기본 proposal은 추가500epoch/branch seeds0,1,2이다.
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label main --stage main --execute
# 중단된 동일 구성만 재개 (새 reward/source/protocol이면 거부)
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label main --stage main --execute --resume
```

C0/C1은 원 checkpoint의 model/optimizer/normalizer/ATS/epoch/frame/RNG를 기존 복원 함수로 로드한다. 새 branch 생성 시 paired continuation seed로 RNG를 재설정한다. 중단 재개 시에는 저장된 RNG를 유지한다. 원 체크포인트에는 full simulator/contact state가 없으므로 **새 episode reset에서 시작**한다. 물리 상태까지 중단 직전과 동일한 resume이라고 주장하지 않는다.

동일 source의 branch seeds0/1/2는 독립 from-scratch 학습 seed가 아니다. source seed가 하나라면 결론도 그 source에 조건부다. 확증하려면 새 독립 fix 학습들의 source에서 반복해야 한다.

원 학습 루프는 `epoch > max_epochs`에서 멈춘다. 새 branch launcher는 target-1을 전달하여 정확히 target에서 종료한다. 예: source1000+500이면 CLI max_epochs1499, 최종 numbered1500. 기존 progress 스크립트는 변경하지 않는다. 최종 checkpoint frame 차이가 `500×2048×32`인지 별도로 검사한다. 저장 간격10에 맞는 추가 epoch만 허용한다.

## 3. C0/C1 평가 및 paired 비교

```bash
C0=/absolute/path/to/branch_main_main_baseline_s0_from1000_plus500/nn/ACTUAL_NAME_00001500.pth
C1=/absolute/path/to/branch_main_main_progress_s0_from1000_plus500/nn/ACTUAL_NAME_00001500.pth
bash experiments/scripts/task_avoidance/evaluate_avoidance.sh \
  --checkpoints "$C0" "$C1" --policy-labels c0,c1 --label branch_s0
bash experiments/scripts/task_avoidance/analyze_avoidance.sh \
  --input experiments/runs/task_avoidance/eval_branch_s0 \
  --left c0 --right c1 \
  --output experiments/runs/task_avoidance/eval_branch_s0/analysis_c0_c1
```

평가에서는 양쪽 정책 모두 **기존 fix 보상**을 받는다. Progress의 높은 total reward와 원 보상을 비교하지 않는다. 나머지 continuation seeds도 각각 같은 방식으로 평가한다. source 자체도 함께 평가하려면 세 checkpoint와 각각 source,c0,c1 label을 명시한다.

## 반환물과 비용

```bash
bash experiments/scripts/progress_reward/collect_results.sh \
  --run experiments/runs/task_avoidance/eval_oldfix_s0 \
  --output /tmp/avoidance-oldfix-s0.tar.gz
# 각 branch run도 같은 방식. 필요한 경우에만 --checkpoints 추가.
```

A/B: 새 학습 없음. 22번 환경 로딩+148step 평가로 시간은 GPU에서 측정해야 한다. 초기 snapshot/history를 저장하므로 기존 JSON-only 평가보다 저장량이 크다. 실행 전에 최소 수 GiB 여유를 확인한다.
C: 과거 book9.70초/epoch에 근거해 추가500epoch C0/C1 한 쌍 약2.7 GPU시간, 3 continuation seeds 총약8.1시간 + 로딩/평가. 추가 checkpoint 대략2.3GiB/run (과거13.81GiB/3000epoch 비례추정); 6run 약14GiB + 로그/평가. 실제 측정값이 아니다. 전 실험을 자동으로 실행하는 명령은 없다.

GPU에서 처음 확인할 것: reward 성분 동등성·실제 초기상태 pairing·repeat probe·20epoch 저장/재개/분기 frame 검증·NaN/보상 스케일. 이를 통과하기 전 본 학습을 실행하지 않는다.
