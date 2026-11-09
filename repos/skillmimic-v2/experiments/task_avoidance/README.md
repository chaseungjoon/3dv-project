# Task Avoidance — v2 실행 안내

고정한 fix에서 **행동 감소(A), 기존 보상의 유인(B), 남은 수행 능력(D), 보상 개입 반응(C)**을 구별한다. 도구와 CPU 검사를 완료한 상태이며 GPU 검증·학습 결과는 아직 없다. PASS는 검사 통과이지 회피 입증이 아니다.

기본 프로토콜은 `experiments/configs/task_avoidance/protocol_v2.json`. 기존 `protocol.json`(v1), fix task/물리/보상/자산과 원 checkpoint는 보존했다. v1의 `lift_attempt`는 실제 상승을 요구하므로 v2에서 `partial_lift`로 이름을 바꾸고 상승 없이도 포착할 수 있는 손목 접근 proxy를 추가했다. v1/v2 지표를 합치지 않는다.

## 0. 기존 GPU 머신에서 준비

```bash
cd /path/to/3dv-project/repos/skillmimic-v2
export PY=/absolute/path/to/existing/isaacgym/python
# ORCHESTRATOR_PY: numpy가 설치된 CPU Python. 기본 python3.
bash experiments/scripts/progress_reward/preflight.sh
SOURCE=/absolute/path/to/place_book_fix_n2048_s0/nn/ACTUAL_NAME_00001000.pth
LATE=/absolute/path/to/place_book_fix_n2048_s0/nn/ACTUAL_NAME_00003000.pth
```

실제 numbered checkpoint 파일명을 확인해서 지정한다. 기존 Linux/Isaac Gym/CUDA 환경, book data, assets, history encoder를 사용한다. 자동 설치/다운로드는 없다. 코드·설정·자산·데이터·encoder·정책 해시와 명령/패키지/GPU를 기록한다.

## 1. 새 학습 없이 실제 task 경로 검사

```bash
bash experiments/scripts/task_avoidance/verify_gpu.sh \
  --checkpoint "$SOURCE" --label checks_v2 --dry-run
bash experiments/scripts/task_avoidance/verify_gpu.sh \
  --checkpoint "$SOURCE" --label checks_v2
```

4환경/약70 control steps. 실제 training task를 실행하되 optimizer 업데이트는 없다. parent reward와 weight=0 동등성, progress 합산, buffer에서 추가 보상 제외, terminal 계측, 부분 reset 시 다른 환경 history/object 보존, reset 전후 거리 캐시를 검사한다. buffer 카운터를 직접 설정하는 검사 fixture이며 연구 결과에 포함하지 않는다. 검사 분기가 실행되지 않으면 INCONCLUSIVE다. 물리 모델 전체 검증은 아니다.

출력 `experiments/runs/task_avoidance/verify_checks_v2/`: `checks/result.json`, `checks.log`, `manifest.json`. 실패 원인을 먼저 해결한다.

## 2. A/B: 기존 정책 비교

```bash
# 사전에 고른 두 정책으로 실행 경로를 먼저 확인한다.
bash experiments/scripts/task_avoidance/evaluate_avoidance.sh \
  --checkpoints "$SOURCE" "$LATE" --label pair_v2 --dry-run
bash experiments/scripts/task_avoidance/evaluate_avoidance.sh \
  --checkpoints "$SOURCE" "$LATE" --label pair_v2
# 전체 곡선: 7개 epoch가 모두 있어야 한다.
bash experiments/scripts/task_avoidance/evaluate_avoidance.sh \
  --run experiments/runs/place_book_fix_n2048_s0 --label curve_v2
```

epoch 750/1000/1500/2000/2500/2750/3000, primary pair 1000→3000. 모든 정책을 첫 evaluation seed에서 재실행한다. 전체는 7정책×3 seeds×32환경 + 반복7×32 =896 episodes/28프로세스, 두 정책은256 episodes/8프로세스다. 모두 순차 실행한다.

같은 seed에서 실제 초기 상태/history/위상/raw input이 다르면 중단한다. 반복 object·wrist·reward·종료가 허용 오차를 벗어나면 paired 결론은 INCONCLUSIVE다. 모든 seed의 결정성을 보장하지는 않는다. `eval_LABEL/analysis/`에 episode·seed별 평균·paired 차이 CSV, 초기 상태 중복 수, SVG 곡선, report를 저장한다. 고유 상태 수는 독립 학습 반복 수가 아니다. primary pair 누락은 INCONCLUSIVE다.

## 3. D: donor-prefix 능력 진단 — 새 학습 없음

```bash
bash experiments/scripts/task_avoidance/run_capability.sh \
  --source "$SOURCE" --candidate "$LATE" --label capability_v2 --dry-run
bash experiments/scripts/task_avoidance/run_capability.sh \
  --source "$SOURCE" --candidate "$LATE" --label capability_v2
```

source(donor)가 실제 simulator에서 0/30/60steps 수행한 뒤 candidate로 정책만 교체한다. 각 프로세스에서 동일 donor prefix를 재실행하고 교체 직전 상태/history/raw input을 비교한다. 물체를 손에 강제로 배치하지 않는다.

|조건|prefix 이후 정책|목적|
|---|---|---|
|direct|donor 계속 실행, reload 없음|기준 궤적|
|reload|같은 donor checkpoint 다시 로드|로드 자체의 영향 검사|
|candidate|후기 정책으로 교체|동일 도달 상태에서 남은 동작 비교|

candidate도 첫 seed에서 반복한다. 3prefix×(3seeds×3조건+1반복)=30프로세스/960 trajectories. 교체 전에 종료된 episode는 분모에서 제외하고 수를 보고한다. 실제 상태를 initial/approach/near_object/holding으로 분류한다. 30/60step이 반드시 잡기 단계라고 가정하지 않는다. donor가 유용한 상태까지 도달하지 못하면 이 검사는 해당 능력을 판단할 수 없다.

출력 `capability_LABEL/analysis/`: controls/eligibility/stage_means/episodes CSV와 report. 모든 self-reload 및 candidate-repeat 검사 통과 시에만 paired_differences CSV를 낸다. **이 결과는 donor 도움 이후 남은 동작의 조건부 능력**이다. 초기부터 전체 과제를 수행할 능력이나 의도를 입증하지 않는다.

## 4. 결과·영상 검토 후 C 분기 학습

접근 proxy는 `PILOT_UNCALIBRATED`다. 결과를 보기 전에 정한 episode 선정 규칙으로 영상과 대조한다. 보정하면 pilot와 분리한 새 프로토콜/평가 집합으로 확증한다. 자동 영상 녹화·수동 판정 일치도 계산은 포함하지 않았다. A/B/D만으로 보상 원인이나 의도적 회피를 확정하지 않는다.

```bash
# source1000 →1020, C0/C1 각 1,310,720 추가 samples.
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label pilot_v2 --stage smoke --extra-epochs 20 --seeds 0 --dry-run
# GPU 검사/평가 확인 후 이 짧은 학습을 실행하기로 결정했을 때:
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label pilot_v2 --stage smoke --extra-epochs 20 --seeds 0 --execute
# 비용과 pilot 결과 검토 후 별도로 실행할 본 실험:
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label main_v2 --stage main --execute
# 중단된 동일 구성만 재개:
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label main_v2 --stage main --execute --resume
```

C0=weight0, C1=weight1. source의 model/optimizer/normalizer/ATS/epoch/frame을 복원한다. 새 branch는 paired continuation RNG seed, 재개는 저장 RNG를 사용한다. full simulator/contact state는 없으므로 fresh episode reset부터 시작한다. 기본 본 실험은 +500epoch×2조건×continuation seeds0/1/2이며 독립 from-scratch 학습 반복이 아니다.

원 루프의 `epoch > max_epochs`에 맞춰 target-1을 전달한다. **목표 numbered checkpoint가 있으면 재학습 없이 epoch/frame/full-state 존재/hash만 감사**한다. 목표 초과는 거부한다. source 대비 실제 frame 차이가 추가 epoch×2048×32인지 확인하며 완료 JSON 없이는 완료 처리하지 않는다. 이전 v1 run은 코드 fingerprint가 다르므로 그대로 resume하지 않는다. 새 label을 쓴다.

## 5. C0/C1도 같은 기존 fix 보상으로 평가

```bash
C0=/absolute/path/to/baseline_run/nn/ACTUAL_NAME_00001500.pth
C1=/absolute/path/to/progress_run/nn/ACTUAL_NAME_00001500.pth
bash experiments/scripts/task_avoidance/evaluate_avoidance.sh \
  --checkpoints "$C0" "$C1" --policy-labels c0,c1 --pair c0 c1 --label branch_s0_v2
```

나머지 continuation seed도 각각 평가한다. `--pair c0 c1`로 자동 분석 대상을 지정한다. C1 shaping을 포함한 학습 total reward와 C0 원 reward를 직접 비교하지 않는다.

## 비용과 결과 전달

verify/A/B/D는 학습이 없지만 GPU가 필요하다. 환경 로딩 시간이 있으므로 실제 시간은 팀원 머신에서 측정한다. 출력에 최소 수 GiB 여유를 잡는다. 과거 기록9.70초/epoch의 단순 비례 추정으로 C smoke 한 쌍 약6.5분, 본 실험6run 약8.1 GPU시간+로딩/평가, 저장량 약14GiB+로그/평가다. 새 측정값은 아니다. 모든 실험을 한꺼번에 실행하는 명령은 없다.

```bash
bash experiments/scripts/progress_reward/collect_results.sh \
  --run experiments/runs/task_avoidance/eval_pair_v2 --output /tmp/avoidance-pair-v2.tar.gz
# capability/verify/각 branch도 같은 방식. 필요할 때만 --checkpoints.
```

같은 label/output은 덮어쓰지 않는다. 실행/분석 오류는 FAILED, 불충분한 진단은 INCONCLUSIVE(exit3)다. 기록을 지우거나 임계값을 낮춰 PASS로 만들지 않는다. 해석 규칙은 [PROTOCOL.md](PROTOCOL.md), 검증 범위는 [VALIDATION.md](VALIDATION.md), 변경 근거는 [CHANGES_V2.md](CHANGES_V2.md)를 참고한다.
