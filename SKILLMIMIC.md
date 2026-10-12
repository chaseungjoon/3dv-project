# SkillMimic-V2 baseline (T09): 상태와 실행 방법

T09 baseline. **SkillMimic-V2** (Yu et al., SIGGRAPH 2025, arXiv 2505.02094, 코드 `Ingrid789/SkillMimic-V2@8e05ad7`)를
RTX 5070 12GB에서 ParaHome clip 세 개(컵, 책, 주전자)에 학습하고, 휴머노이드가 물체를 **쥐고 들어 올리는지** 잰다.
규칙, 코드 변경, 컵 진단은 `SKILLMIMIC_NOTES.md`. 결과는 `BASELINE_SKILLMIMIC*.md` (자동 생성).

**왜 SkillMimic-V2인가 (2026-10-10 팀 결정).** WristMimic(이전 T09 baseline)은 이 GPU에서 주전자를 쥐고 드는 순간을 넘지 못해 0%였다
(`BASELINE_WRISTMIMIC.md`). SkillMimic-V2는 과제 안내서에서 WristMimic의 비교 대상이고, 같은 ParaHome 데이터와 Isaac Gym을 쓰며,
논문에서 세 clip 모두 성공률 100%이고, 논문과 같은 2048 env가 12GB에 들어간다.

## 1. 현재 상태 (2026-10-12, 최종)

결과 문서: 1차(업스트림 환경) `BASELINE_SKILLMIMIC.md`, **2차(고친 환경) `BASELINE_SKILLMIMIC_2.md`**. 해석과 다음 연구 주제는 `PROPOSAL.md`.

| clip | 내용 | run | 상태 | 들어올림 성공 | 과제 성공 |
|---|---|---|---|---|---|
| 컵 `drink_cup` | 식탁의 컵을 한 손으로 들어 입으로 가져갔다가 내려놓기 (180 frame) | `drink_cup_ours_n2048_s0` (업스트림 환경) | 끝남 | 0% (최대 8 cm) | 0% |
| | | `drink_cup_fix_n2048_s0` (**고친 환경**) | 끝남 (3000 epoch) | **75%** (최고 88%: epoch 2200, 2400, 2500, 2710) | 0% |
| 책 `place_book` | 책상의 책을 두 손으로 집어 왼손으로 들고 1.6 m 옮기기 (150 frame) | `place_book_ours_n2048_s0` (업스트림 환경) | 끝남 | 0% (최대 4 cm) | 0% |
| | | `place_book_fix_n2048_s0` (**고친 환경**) | 끝남 (3000 epoch) | **38%** (최고 100%: epoch 1000) | 0% |
| 주전자 `place_kettle` | 주전자를 들어 옮기기 (100 frame), T09와 같은 물체 | `place_kettle_ours_n2048_s0` | epoch 60에서 멈춤, 보류 | - | - |

수치는 결정적 8 env, frame 2 시작이다. 고친 환경 run의 특징은 다음과 같다.

| | 컵 `fix` | 책 `fix` |
|---|---|---|
| 들기가 생긴 시점 | epoch 2200에 갑자기 (2100까지 0%) | epoch 750 |
| 과제 성공 0%의 이유 | 내려놓기가 늦다 (frame 95~125 물체 오차 41~57 cm). 들기 구간부터 시연 궤적과 약 20 cm 어긋남 | 잡기와 안정성이 맞바뀐다. 옮기던 epoch 1000~1500은 88~100%가 넘어졌다. 3000은 넘어짐 12%지만 책을 시연(145 cm)의 33 cm만 옮긴다 |
| 확률적 32 env / + 물체 교란 | 59% / 3% | 28% / 19% |
| 마지막 체크포인트가 최선인가 | 아니다 (2500, 2710이 88%) | 아니다 (1000이 100%) |
| 최대 VRAM, 학습 시간 | 11.5 GB, 9.1 h | 9.3 GB, 8.1 h |

업스트림 환경에는 버그가 있었다.
- 가구 충돌 형상이 거칠어 물체가 높게 놓인다. 컵은 reset마다 튕겨 나간다.
- 물체가 너무 무겁다 (컵 1.49 kg, 책 7.1 kg).
- history 입력 버그가 있고, 위상 정보가 없다.

고친 환경(`fix`)에서는 두 물체 모두 들기를 배웠다: `SKILLMIMIC_NOTES.md` 4절. 같은 GPU와 같은 학습량이다. 주전자는 보류다.

## 2. 실행

모든 명령은 `repos/skillmimic-v2`에서. 학습은 `tmux` 안에서 권장. **GPU 작업은 한 번에 하나만.**

```bash
cd ~/code/3dv-project/repos/skillmimic-v2
bash run_all.sh          # 2차 전체: 컵 -> 평가 -> 책 -> 평가 -> 결과 문서 (2026-10-12 끝남, BASELINE_SKILLMIMIC_2.md). 다시 실행하면 끝난 것은 건너뛰고 새 번호 문서만 만든다
bash run_cup.sh          # 컵만 (고친 환경): 학습 3000 epoch -> 평가
bash run_book.sh         # 책만 (고친 환경): 학습 3000 epoch -> 평가. 약 10시간
bash run_kettle.sh       # 주전자 (업스트림 환경, 보류), epoch 60부터 이어서
bash write_results.sh    # 결과 문서만 (GPU 안 씀, 언제든)
```

- `run_*.sh` = 학습 3000 epoch (2048 env, 1.97억 샘플, 논문의 20%) → 평가 (250 epoch마다 학습 곡선 + 최종 체크포인트 3가지). 결과 문서는 쓰지 않는다.
- `write_results.sh`는 기존 문서를 **덮어쓰지 않는다**: 프로젝트 루트에 `BASELINE_SKILLMIMIC_2.md`, `_3.md`, ... (가장 큰 번호 + 1). 그림도 문서마다 따로.
- `run_all.sh`: 컵, 책(`CLIPS`로 바꿀 수 있음)을 차례로 하고 마지막에 `write_results.sh`까지 한다. 끝난 학습/평가는 건너뛴다.
  다른 SkillMimic 작업이 돌고 있으면 끝날 때까지 기다린다. 한 clip이 끝내 실패해도 다음 clip과 결과 문서는 진행한다. 전체 출력은 `experiments/logs/run_all_*.log`.

### 끊겼을 때

**같은 명령을 다시 실행하면 된다** (Ctrl-C, 터미널 종료, 정전 모두).

- 학습은 마지막 체크포인트(10 epoch마다, 약 2분)부터 그대로 이어진다: epoch, optimizer, 정규화 통계, ATS 상태, 난수 상태까지 복원.
- 평가는 이미 한 체크포인트를 건너뛴다. 학습이 끝난 run이면 남은 평가만 한다.
- 학습이 죽으면 (예: PhysX GPU 메모리 할당 실패) 1분 뒤 자동으로 2번까지 이어서 한다. Ctrl-C는 바로 멈춘다.
- 저장은 임시 파일에 쓰고 이름을 바꾸므로 저장 중에 끊겨도 깨진 체크포인트가 없다.

### 옵션 (환경변수)

| 변수 | 기본 | 뜻 |
|---|---|---|
| `EPOCHS` | 3000 | 총 epoch. 끝난 run에 더 큰 값을 주면 거기서부터 더 학습 (`EPOCHS=6000 bash run_cup.sh`) |
| `TRAIN_RETRIES` | 2 | `run_*.sh`: 학습이 죽었을 때 자동으로 이어 하는 횟수 |
| `RETRIES` | 2 | `run_all.sh`: clip 하나가 죽었을 때 다시 하는 횟수 |
| `CLIPS` | `drink_cup place_book` | `run_all.sh`가 할 clip |
| `NUM_ENVS` | 2048 | 메모리 부족일 때만 1536 (run 이름이 바뀐다) |
| `SEED` | 0 | seed 추가 실험 |
| `EVAL_EVERY` | 250 | 학습 곡선 평가 간격 (epoch) |

### 학습 중 보기

```bash
tail -f experiments/runs/drink_cup_fix_n2048_s0/train.log   # 전체 출력 (epoch_num, mean_rewards)
tensorboard --logdir experiments/runs                       # 보상, episode 길이
bash write_results.sh                                       # 진행 상태 문서
```

중간 체크포인트를 보고 싶으면 Ctrl-C로 멈추고 `bash experiments/scripts/eval_run.sh drink_cup_fix_n2048_s0 curve 1000` 후 다시 `run_cup.sh`.

## 3. 결과와 파일 위치 (`repos/skillmimic-v2/` 기준)

| 무엇 | 위치 |
|---|---|
| 결과 문서 | 프로젝트 루트 `BASELINE_SKILLMIMIC.md`(1차: 컵·책 업스트림 환경), `BASELINE_SKILLMIMIC_2.md`(2차: 컵·책 고친 환경, 최종), 이후 `_3.md`, ... |
| 체크포인트, `train.log`, `gpu.csv`, `run_meta.json`(끊긴 구간 기록), TensorBoard | `experiments/runs/<run>/` (git 제외, 체크포인트 10 epoch마다 모두 유지, run당 약 13 GB) |
| 체크포인트별 평가 (env별 지표, frame별 궤적) | `experiments/results/eval/<det\|stoch\|stoch_perturb>/<run>/` |
| 표, 그림 | `experiments/results/tables/*.csv` (최신), `experiments/results/figures/baseline_curves[_N].png` |
| 컵 진단 때의 비교 평가 | `experiments/probe/results/` |

## 4. 개별 명령

```bash
bash experiments/scripts/train.sh drink_cup fix 3000                 # 학습만 (자동 이어 학습). method: fix (컵, 책) | ours (주전자, 업스트림 비교)
bash experiments/scripts/eval_run.sh <run> final                     # 마지막 체크포인트: det + stoch + 교란
bash experiments/scripts/eval_run.sh <run> curve 250,500,750         # 학습 곡선용 det
.venv/bin/python experiments/tools/ckpt.py latest experiments/runs/<run>        # 다음에 이어 학습할 체크포인트
.venv/bin/python experiments/tools/evaluate.py --run_dir experiments/runs/<run> --viewer --tag viewer --force   # 눈으로 보기 (화면, env 1개)
.venv/bin/python experiments/probe/env_probe.py --clip drink_cup --method fix   # 정책 없이 환경 점검 (NOTES 4절), --clip place_book도 됨
```

처음 설치 (이 컴퓨터는 끝남): `bash experiments/scripts/setup.sh` (uv 환경, ParaHome 장면 s6/s10/s22, history encoder 약 6분),
`bash experiments/scripts/pipeline_check.sh` (약 15분, 끊고 다시 이어지는지까지 확인). Isaac Gym 자체는 `isaacgym-env/README.md`.

## 5. 문제가 생기면

| 증상 | 원인 / 해결 |
|---|---|
| `CUDA out of memory`, `PxgCudaDeviceMemoryAllocator fail` | 2048 env는 시작 때 9.3~10.8 GB, 접촉이 늘면 11.5 GB까지. 자동으로 2번 이어 한다. 계속 죽으면 브라우저 등 GPU 쓰는 프로그램을 끄고, 그래도 안 되면 `NUM_ENVS=1536` |
| `history encoder 없음`, `ParaHome 원본 링크 없음` | `bash experiments/scripts/setup.sh` |
| `[ckpt] cannot load ...` | 깨진 체크포인트는 건너뛰고 그 전 것에서 이어간다. 정상 |
| `shape mismatch ... 1025 ... 1028` | asset과 task가 안 맞음. `run.py`를 직접 부르지 말고 `train.sh`를 쓴다 |
| 시작 후 1~2분 멈춘 듯 보임 | 2048 env 생성 + 충돌 형상 분해. 정상 |
| `fix` 평가에서 넘어짐이 많음 | 학습 중 물체를 놓치면 episode를 끝내서 그 뒤 frame을 거의 겪지 않는다. 들기를 배우기 전에는 정상 (컵은 epoch 2200 전까지 그랬다) |
