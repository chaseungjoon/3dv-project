# PAPER vs FIX: 약 10억 samples 실험 실행 계획

이 문서는 `place_book` 공개 구성 `ours`와 수정 구성 `fix`를 같은 학습량으로 비교하기 위한 실행 체크리스트다. 모든 명령은 Linux GPU 머신의 `repos/skillmimic-v2`에서 실행한다.

## 1. 연구 질문과 주장 범위

> 공개 SkillMimic-V2 구성(`ours`)과 수정 환경(`fix`)을 약 10억 samples로 학습했을 때 학습 안정성, 책 조작 성능, 물체 교란 일반화에 어떤 차이가 나타나는가?

`fix`는 충돌 형상, 책 질량, history layout, phase observation, 물체 조기 종료가 함께 바뀐 묶음이다. 차이가 나더라도 특정 수정 하나가 아니라 **fix 전체 패키지의 효과**로 보고한다. 저자의 내부 자산·미공개 설정까지 동일하다고 주장하지 않고 **공개 자료 기반 paper-budget reproduction**이라고 부른다.

## 2. 사전에 고정할 실험

|조건|clip|method|training seeds|epochs|samples/seed|
|---|---|---|---|---:|---:|
|PAPER|`place_book`|`ours`|0, 1, 2|15,260|1,000,079,360|
|FIX|`place_book`|`fix`|0, 1, 2|15,260|1,000,079,360|

```text
2048 environments × horizon 32 = 65,536 samples/epoch
65,536 × 15,260 epochs = 1,000,079,360 samples
```

총 6개 정책을 처음부터 독립적으로 학습한다. `ours`와 `fix`는 observation·task·asset이 다르므로 checkpoint를 공유하거나 서로 이어 학습하지 않는다.

고정 설정:

```text
num_envs                 2048
horizon_length           32
episode_length           60
gamma                    0.99
learning_rate            2e-5
minibatch_size           16384
mini_epochs              6
state_init_random_prob   0.1
reweight_alpha           1.0
history_length           60
history_embedding_size   3
```

## 3. 시작 전 체크리스트

- [ ] GPU 작업은 한 번에 하나만 실행한다.
- [ ] 가능하면 모든 run에 같은 GPU를 사용한다.
- [ ] 최소 500GB의 여유 공간을 확보한다.
- [ ] motion, assets, history encoder가 준비돼 있다.
- [ ] 시작 commit과 파일 hash를 기록한다.
- [ ] 실험 도중 code/config/data/encoder를 변경하지 않는다.
- [ ] 평가 checkpoint·seed·지표를 결과 보기 전에 고정한다.

```bash
cd /absolute/path/to/3dv-project
git status --short
git rev-parse HEAD

cd repos/skillmimic-v2
export PY=/absolute/path/to/isaacgym/environment/bin/python

"$PY" -c "import isaacgym; import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
df -h .
```

필요하면 최초 한 번 실행한다.

```bash
bash experiments/scripts/setup.sh
bash experiments/scripts/pipeline_check.sh
```

## 4. 결과 폴더 분리

기존 3000 epoch 결과와 섞이지 않게 별도 root를 사용한다.

```text
PAPER runs     experiments/runs_paper_public/
PAPER results  experiments/results_paper_public/
FIX runs       experiments/runs_paper_fix/
FIX results    experiments/results_paper_fix/
```

기존 `experiments/runs/place_book_*`에서 resume하지 않는다.

## 5. 20 epoch smoke test

PAPER:

```bash
PY="$PY" RUNS_DIR=experiments/runs_paper_public \
RESULTS_DIR=experiments/results_paper_public NUM_ENVS=2048 \
bash experiments/scripts/train.sh place_book ours 20 0
```

FIX:

```bash
PY="$PY" RUNS_DIR=experiments/runs_paper_fix \
RESULTS_DIR=experiments/results_paper_fix NUM_ENVS=2048 \
bash experiments/scripts/train.sh place_book fix 20 0
```

확인:

```bash
find experiments/runs_paper_public/place_book_ours_n2048_s0/nn -name '*_00000020.pth'
find experiments/runs_paper_fix/place_book_fix_n2048_s0/nn -name '*_00000020.pth'
```

- [ ] 두 checkpoint가 생성됐다.
- [ ] `run_meta.json`, `train.log`, `gpu.csv`가 생성됐다.
- [ ] NaN, traceback, CUDA OOM이 없다.
- [ ] PAPER=`ours`, FIX=`fix`로 기록됐다.
- [ ] 재실행하면 epoch 20에서 정상 resume한다.

smoke 중 코드를 수정했다면 기존 smoke를 본 결과에 포함하지 않고 새 출력 root에서 다시 시작한다.

## 6. 15,260 epoch 본 학습

공통 설정:

```bash
export PY=/absolute/path/to/isaacgym/environment/bin/python
export NUM_ENVS=2048
export EPOCHS=15260
export EVAL_EVERY=500
export TRAIN_RETRIES=2
```

한 GPU에서 다음 순서로 교차 실행한다.

```text
PAPER s0 → FIX s0 → FIX s1 → PAPER s1 → PAPER s2 → FIX s2
```

PAPER s0:

```bash
RUNS_DIR=experiments/runs_paper_public RESULTS_DIR=experiments/results_paper_public \
SEED=0 bash experiments/scripts/run_baseline.sh place_book ours
```

FIX s0:

```bash
RUNS_DIR=experiments/runs_paper_fix RESULTS_DIR=experiments/results_paper_fix \
SEED=0 bash experiments/scripts/run_baseline.sh place_book fix
```

FIX s1:

```bash
RUNS_DIR=experiments/runs_paper_fix RESULTS_DIR=experiments/results_paper_fix \
SEED=1 bash experiments/scripts/run_baseline.sh place_book fix
```

PAPER s1:

```bash
RUNS_DIR=experiments/runs_paper_public RESULTS_DIR=experiments/results_paper_public \
SEED=1 bash experiments/scripts/run_baseline.sh place_book ours
```

PAPER s2:

```bash
RUNS_DIR=experiments/runs_paper_public RESULTS_DIR=experiments/results_paper_public \
SEED=2 bash experiments/scripts/run_baseline.sh place_book ours
```

FIX s2:

```bash
RUNS_DIR=experiments/runs_paper_fix RESULTS_DIR=experiments/results_paper_fix \
SEED=2 bash experiments/scripts/run_baseline.sh place_book fix
```

중단되면 code·환경변수·명령을 바꾸지 않고 해당 명령을 다시 실행한다. 최신 full checkpoint에서 resume한다. 실패 기록은 삭제하지 않는다.

## 7. 완료 확인

```bash
find experiments/runs_paper_public -type f -path '*/nn/*_00015260.pth'
find experiments/runs_paper_fix -type f -path '*/nn/*_00015260.pth'
```

각 조건에서 3개씩 총 6개가 있어야 한다. 파일명만 보지 말고 checkpoint 내부 epoch/frame과 SHA-256도 기록한다. 기대 sample 수는 run당 `1,000,079,360`이다.

## 8. 사전 지정 평가

평가 checkpoint:

```text
500, 750, 1000, 1500, 2000, 2500, 3000,
4000, 5000, 7500, 10000, 12500, 15000, 15260
```

- Primary checkpoint: 최종 epoch 15,260
- Secondary: 전체 learning curve
- Exploratory: 조건별 best checkpoint

best checkpoint만 골라 PAPER와 FIX를 비교하지 않는다.

Primary metrics:

```text
task_success under stochastic initialization
task_success under object perturbation
goal progress 또는 final object error
```

Secondary metrics:

```text
lift success, partial lift, fall rate, object tracking error,
evaluation reward, 첫 성공 epoch, 최고 성능과 최종 성능의 차이
```

독립 통계 단위는 environment episode가 아니라 **training seed 3개**다.

## 9. checkpoint 평가

PAPER s0:

```bash
RUNS_DIR=experiments/runs_paper_public RESULTS_DIR=experiments/results_paper_public \
bash experiments/scripts/eval_run.sh place_book_ours_n2048_s0 curve \
500,750,1000,1500,2000,2500,3000,4000,5000,7500,10000,12500,15000,15260

RUNS_DIR=experiments/runs_paper_public RESULTS_DIR=experiments/results_paper_public \
bash experiments/scripts/eval_run.sh place_book_ours_n2048_s0 final
```

FIX s0:

```bash
RUNS_DIR=experiments/runs_paper_fix RESULTS_DIR=experiments/results_paper_fix \
bash experiments/scripts/eval_run.sh place_book_fix_n2048_s0 curve \
500,750,1000,1500,2000,2500,3000,4000,5000,7500,10000,12500,15000,15260

RUNS_DIR=experiments/runs_paper_fix RESULTS_DIR=experiments/results_paper_fix \
bash experiments/scripts/eval_run.sh place_book_fix_n2048_s0 final
```

seed 1·2도 run 이름만 바꿔 같은 checkpoint를 평가한다.

## 10. 최종 checkpoint 반복 평가

각 training seed의 epoch 15,260을 evaluation seed 101·102·103으로 평가한다. 모든 조건에 같은 evaluation seed를 사용한다.

PAPER s0 예시:

```bash
for eval_seed in 101 102 103; do
  "$PY" experiments/tools/evaluate.py \
    --run_dir experiments/runs_paper_public/place_book_ours_n2048_s0 \
    --epochs 15260 --mode stoch --num_envs 32 \
    --seed "$eval_seed" --tag "s$eval_seed" \
    --out_dir experiments/results_paper_public/eval

  "$PY" experiments/tools/evaluate.py \
    --run_dir experiments/runs_paper_public/place_book_ours_n2048_s0 \
    --epochs 15260 --mode stoch --num_envs 32 --perturb \
    --seed "$eval_seed" --tag "s$eval_seed" \
    --out_dir experiments/results_paper_public/eval
done
```

FIX는 다음 경로로 바꿔 동일하게 실행한다.

```text
--run_dir experiments/runs_paper_fix/place_book_fix_n2048_s0
--out_dir experiments/results_paper_fix/eval
```

training seed 1·2에도 반복한다. perturb는 z축 회전 ±45도, xy 최대 10cm다.

## 11. 결과표와 판정

|training seed|PAPER success|FIX success|PAPER perturb|FIX perturb|PAPER goal progress|FIX goal progress|
|---|---:|---:|---:|---:|---:|---:|
|0|||||||
|1|||||||
|2|||||||
|평균 ± 표준편차|||||||

FIX 개선을 지지하려면:

- [ ] 3개 training seed 중 최소 2개에서 FIX 최종 성능이 높다.
- [ ] 평균 task success와 goal/object 지표가 같은 방향이다.
- [ ] 일반 초기화와 perturb 평가에서 방향이 유지된다.
- [ ] 반복 평가 변동보다 조건 차이가 크다.
- [ ] fall 감소만으로 성공률 개선을 설명하지 않는다.

다음이면 `INCONCLUSIVE`다.

- seed마다 우열 방향이 다르다.
- 한 seed만 큰 효과를 보인다.
- 반복 평가 변동이 조건 차이와 비슷하다.
- final은 비슷하고 best checkpoint에서만 차이가 난다.
- 학습 실패나 checkpoint 누락이 있다.

허용 가능한 결론:

> 공개 구성과 비교해 fix 전체 패키지가 같은 약 10억 sample 예산에서 학습 안정성 또는 책 조작 성능을 개선했다/개선하지 못했다.

개별 질량·충돌·phase 수정의 효과는 이 실험만으로 주장하지 않는다.

## 12. 예상 비용

```text
한 run 약 41시간
6 runs 약 246 GPU시간
단일 GPU에서 약 10일 이상

run당 checkpoint 약 65~70GB
6 runs 약 390~420GB
권장 여유 공간 최소 500GB
```

첫 full run에서 실제 시간과 용량을 다시 측정한다. 결과를 본 뒤 seed 수나 종료 epoch를 임의로 줄이지 않는다. 예산을 변경해야 하면 변경 전에 새 계획과 이유를 기록한다.

## 13. 최종 제출물

- [ ] Git commit과 시작 시 `git status`
- [ ] config/data/asset/history-encoder hash
- [ ] 6개 `run_meta.json`
- [ ] 최종 checkpoint epoch/frame/hash
- [ ] training seed별 learning curve
- [ ] evaluation seed별 JSON/NPZ 원자료
- [ ] stochastic 및 perturb 결과표
- [ ] seed별 값과 평균±표준편차
- [ ] 실패·재개·OOM 기록
- [ ] 실행 명령, GPU, 실제 시간
- [ ] 사실·추정·미확인 사항을 구분한 보고서

## 실행 순서 요약

```text
1. commit·환경·파일 hash 고정
2. PAPER/FIX 각각 20 epoch smoke
3. PAPER/FIX × seeds 0,1,2를 15,260 epoch 학습
4. 최종 checkpoint와 10억 sample 확인
5. 사전 지정 checkpoint learning curve 평가
6. 최종 checkpoint를 eval seeds 101,102,103으로 반복 평가
7. stochastic 및 perturb 결과를 training seed 단위로 비교
8. final 성능을 primary, best checkpoint를 exploratory로 보고
9. fix 전체 패키지 효과까지만 결론
```
