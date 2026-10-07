# T09 WristMimic 실험 실행 가이드

과제 **T09 Wrist-guided humanoid manipulation**의 baseline 학습, 손목 변형 실험, 평가, 발표 자료 생성을 위한 실행 가이드다.
baseline은 **WristMimic**이고, 전체 계획은 `3dv-project/PROPOSAL_T09.md`에 있다.

- 무엇을 어떻게 재는지(규칙, 지표 정의, 가설 판정): [`docs/PROTOCOL.md`](docs/PROTOCOL.md)
- 코드에서 확인한 사실과 원본 대비 변경점(발표 슬라이드용): [`docs/CODE_NOTES.md`](docs/CODE_NOTES.md)

> **모든 명령은 저장소 루트 `~/code/3dv-project/repos/t09-wristmimic`에서 실행한다.**
> 긴 학습은 SSH가 끊겨도 살아 있도록 `tmux`(또는 `screen`) 안에서 돌리는 것을 권장한다.

---

## 0. 한눈에 보기: 실행 순서

| 단계 | 명령 | GPU 시간 (실측 기반) | 끝나면 확인할 것 |
|---|---|---|---|
| 점검 (완료) | `bash experiments/scripts/pipeline_check.sh` | 약 5분 | "파이프라인 점검 통과" |
| **Phase 1** 기본 설정 긴 학습 | `bash experiments/scripts/phase1_baseline_long.sh` | 6000 epoch ≈ **10.3시간** + 평가 약 10분 | `report.sh`의 "제안 T" → 학습 예산 T 결정 |
| **Phase 2** 기본 설정 seed 1, 2 | `BUDGET=<T> bash experiments/scripts/phase2_baseline_seeds.sh` | 2 × T | baseline 표 (seed 3개) |
| **Phase 3** 최소 비교 | `BUDGET=<T> bash experiments/scripts/phase3_minimal_compare.sh` | 4 × T | 손목 제약 없음 / 임계값 2단계 |
| **Phase 4** 연구 질문 변형 | `BUDGET=<T> bash experiments/scripts/phase4_research_questions.sh` | 6 × T | RQ1(회전 임계값), RQ2(가중치, 창 시점) |
| **Phase 5** 주장할 변형 seed 추가 | `BUDGET=<T> bash experiments/scripts/phase5_extra_seeds.sh <변형...>` | 2 × T × 변형 수 | 가설 판정 (seed 3개) |
| 리포트 (언제든) | `bash experiments/scripts/report.sh <T>` | 수 초 (GPU 안 씀) | `experiments/results/report/REPORT.md` |
| 실패 영상 | `bash experiments/scripts/record_video.sh <체크포인트>` | 약 1분 (화면 필요) | `experiments/results/videos/*.mp4` |

각 Phase 스크립트는 학습이 끝날 때마다 그 run을 **자동 평가**한다 (끄려면 `AUTO_EVAL=0`).
Phase 2~4의 학습은 서로 독립이라 순서를 바꿔도 된다. GPU가 하나라서 **동시에 두 학습을 돌리지 않는다** (1024 env 학습 1개가 약 8 GB를 쓴다).

---

## 1. 폴더 구조

```
experiments/
├── README.md                  이 문서
├── docs/
│   ├── PROTOCOL.md            실험 규칙, 지표 정의, 예산 규칙, 가설 판정
│   └── CODE_NOTES.md          코드 분석, 원본 대비 변경, 함정
├── configs/
│   ├── variants.yaml          ★ 모든 변형의 정의 (여기만 고친다)
│   └── env/                   make_configs.py가 생성한 학습/평가 설정 (직접 고치지 않음)
│       ├── default.yaml ...   변형별 학습 설정 11개
│       └── eval_default.yaml  모든 run을 채점하는 평가 설정 (기본 손목 임계값)
├── tools/
│   ├── make_configs.py        variants.yaml → configs/env/*.yaml (손목 외 키 변경 시 에러)
│   ├── evaluate.py            체크포인트 평가 (env 한 번 생성, 여러 체크포인트)
│   ├── metrics.py             지표 계산 (env를 읽기만 함)
│   └── aggregate.py           표, 그림, REPORT.md 생성
├── scripts/                   실행 스크립트 (아래 각 절)
├── runs/                      학습 결과 (git 제외, run 하나에 약 1.3 GB)
│   └── <변형>_s<seed>_<월일-시분초>/
│       ├── nn/                체크포인트: _000XXXXX.pth (500 epoch마다), _latest.pth (매 epoch)
│       ├── summaries/         TensorBoard
│       ├── source/            설정 snapshot, git diff, 재현 명령
│       ├── train.log          학습 출력 전체
│       ├── gpu.csv            30초마다 GPU 메모리/사용률/온도/전력
│       └── run_meta.json      변형, seed, 예산, 시작/종료, 총 시간, 최대 VRAM
└── results/                   (git에 포함)
    ├── eval/det/<run>/        결정적 평가: 체크포인트별 json(지표) + npz(프레임별 시계열)
    ├── eval/stoch/<run>/      확률적 평가
    ├── report/                REPORT.md, csv, fig_*.png  ← 발표 자료
    └── videos/                실패 사례 mp4
```

## 2. 사전 점검 (2026-10-04 완료)

`bash experiments/scripts/pipeline_check.sh` (약 5분, 20 epoch 학습 포함)를 실행해 통과했다. 확인한 것:

| 항목 | 결과 |
|---|---|
| 설정 11개 + 평가 설정이 base와 손목 키만 다름 | OK |
| 학습 실행, 체크포인트 저장, TensorBoard 기록 | OK |
| **epoch당 시간** | **6.1~6.2초** (1024 env, 약 5,800 env-step/s) → 1000 epoch ≈ 1.7시간 |
| 최대 GPU 메모리 | 8.0~8.2 GB / 12 GB |
| 평가 재현성 (8-env 결정적 평가를 별도 프로세스로 2번) | 허용 오차 안 (OK) |
| 원본 `run.py --test --test_no_reset` 결과가 우리 평가 분포 안에 있음 | OK |
| 실패 플래그가 env의 terminate 플래그와 일치, 원인이 설명 안 되는 실패 0건 | OK |
| viewer 녹화 → mp4 | OK |

**중요한 발견:** GPU PhysX는 비트 단위로 재현되지 않는다. 같은 체크포인트, 같은 결정적 행동이라도 프레임 6쯤부터 1e-5 m 수준의 차이가 생긴다. 그래서 결정적 평가도 env 8개로 반복한다 (PROTOCOL.md 4.3).

코드나 설정을 고친 뒤에는 이 점검을 다시 돌린다. 결과는 `experiments/runs/_checks`, `experiments/results/_checks`에만 쓰인다.

## 3. 학습

### 3.1 단일 학습

```bash
bash experiments/scripts/train.sh <변형> <seed> <epoch 수>
# 예) bash experiments/scripts/train.sh default 0 3000
```

- 변형 이름은 `experiments/configs/env/`의 파일 이름 (`default`, `reset_off`, `wrist_off`, `pos2_x0.5`, `pos2_x2`, `rot2_x0.5`, `rot2_x2`, `gwp_30`, `gwp_140`, `win_early5`, `win_late5`).
- epoch 수는 **500의 배수**로 준다. 체크포인트가 500 epoch마다 저장되므로 마지막 체크포인트가 정확히 그 epoch가 된다 (비교에 이 체크포인트를 쓴다).
- 환경변수로 바꿀 수 있는 것: `NUM_ENVS`(기본 1024), `SCENE`, `ROBOT`, `RUN_NAME`, `RESUME_FROM`. 바꾸면 그 run은 다른 run과 비교할 수 없다는 점에 주의.

### 3.2 Phase 1: 기본 설정 긴 학습 (가장 먼저)

```bash
tmux new -s t09
cd ~/code/3dv-project/repos/t09-wristmimic
bash experiments/scripts/phase1_baseline_long.sh          # EPOCHS_LONG=6000 (기본)
```

- 약 10.3시간. 500 epoch마다 체크포인트 12개가 생긴다.
- 끝나면 자동으로 모든 체크포인트를 평가하고, 다음을 실행한다:

```bash
bash experiments/scripts/report.sh
```

- `REPORT.md` 1절의 **제안 T**와 표를 보고 학습 예산 T를 정한다 (규칙: PROTOCOL.md 5.2). 정한 값과 이유를 PROTOCOL.md 7절 변경 기록에 적는다.
- Phase 1 run의 epoch T 체크포인트가 그대로 "기본 설정 seed 0"의 결과가 된다 (다시 학습하지 않음).
- **6000 epoch가 너무 길면** `EPOCHS_LONG=4000` 등으로 줄인다. 중간에 멈춰도 그때까지의 체크포인트로 곡선을 그릴 수 있다 (6절).

### 3.3 Phase 2~5

```bash
BUDGET=3000 bash experiments/scripts/phase2_baseline_seeds.sh      # default seed 1, 2
BUDGET=3000 bash experiments/scripts/phase3_minimal_compare.sh     # reset_off, pos2_x2, wrist_off, pos2_x0.5
BUDGET=3000 bash experiments/scripts/phase4_research_questions.sh  # rot2_x2, rot2_x0.5, gwp_30, gwp_140, win_early5, win_late5
BUDGET=3000 bash experiments/scripts/phase5_extra_seeds.sh pos2_x2 rot2_x2   # 주장할 변형에 seed 1, 2
```

(3000은 예시. Phase 1에서 정한 T를 넣는다.)

일부만 돌리려면 `JOBS`로 고른다:

```bash
JOBS="reset_off:0 pos2_x2:0" BUDGET=3000 bash experiments/scripts/phase3_minimal_compare.sh
# 또는 큐를 직접:  bash experiments/scripts/queue.sh 3000 reset_off:0 pos2_x2:0
```

## 4. 시간 예산

GPU 1장, 순차 실행. epoch당 6.15초 기준이다 (평가는 run당 수 분이라 제외).

| 묶음 | 학습 수 | T = 2000 | T = 3000 | T = 4000 |
|---|---|---|---|---|
| Phase 1 (6000 epoch 고정) | 1 | 10.3 h | 10.3 h | 10.3 h |
| Phase 2 | 2 | 6.8 h | 10.3 h | 13.7 h |
| Phase 3 | 4 | 13.7 h | 20.5 h | 27.3 h |
| Phase 4 | 6 | 20.5 h | 30.8 h | 41.0 h |
| Phase 5 (변형 2개) | 4 | 13.7 h | 20.5 h | 27.3 h |
| **합계** | 17 | **65 h** | **92 h** | **120 h** |

**중간발표 최소선** (PROPOSAL_T09.md 5절): Phase 1 + Phase 2 + Phase 3의 앞 두 개(`reset_off`, `pos2_x2`) + 실패 영상 1개.
T = 3000이면 10.3 + 10.3 + 10.3 ≈ **31시간**.

## 5. 모니터링

학습 중에는 다른 터미널에서:

```bash
# 학습 로그 (epoch, 평균 보상, fps, epoch 시간)
tail -f experiments/runs/<run>/train.log | grep --line-buffered epoch_num

# TensorBoard (브라우저에서 http://localhost:6006)
.venv/bin/tensorboard --logdir experiments/runs --port 6006

# GPU
watch -n 5 nvidia-smi
```

TensorBoard에서 볼 것 (wandb를 끄고 쓰므로 원본이 wandb로만 보내던 지표를 `wm/` 아래에 같이 기록한다):

| 태그 | 의미 | 정상 추세 |
|---|---|---|
| `episode_lengths/iter` | 학습 에피소드 평균 길이 (프레임) | 오르다가 시퀀스 길이(약 154) 근처로 |
| `wm/info/episode_progress` | 에피소드가 시퀀스의 몇 %까지 갔나 | 0 → 1 쪽으로 |
| `wm/info/episode_progress_ratio_90` | 90% 이상 간 에피소드 비율 | 성공 정책이면 0 → 높게 |
| `wm/info/contact_progress` | 경과 프레임 / 첫 접촉 프레임 | 1을 넘으면 잡기 구간을 통과하기 시작 |
| `rewards0/iter` | 평균 에피소드 보상 | 증가 |
| `wm/reward_body/wrist_rwp`, `wrist_rwr` | 손목 위치/회전 보상 (1이 최고) | 증가 |
| `wm/reward_object/rop` | 물체 위치 보상 | 증가 |
| `wm/time/wall_clock_s` | 경과 시간 | |

**이상 신호**

| 증상 | 의미 / 대응 |
|---|---|
| `invalid observation` 예외로 종료 | 시뮬레이션 발산 (NaN). `_latest.pth`에서 이어 학습 (6절), 반복되면 기록하고 팀 논의 |
| `episode_lengths`가 수천 epoch 동안 10 이하 | 시작하자마자 termination. `train.log`에서 확인, 장면/설정 점검 |
| fps가 평소(약 5,800 env-step/s)보다 크게 낮음 | 다른 GPU 프로세스가 있는지 `nvidia-smi` 확인 |
| `CUDA out of memory`, `PxgCudaDeviceMemoryAllocator` | 다른 GPU 작업 종료. 그래도 나면 `NUM_ENVS=512` (단, 비교 불가 run이 됨) |
| 이어 학습 시 `Failed to restore from checkpoint` | 체크포인트 경로 오류. 처음부터 다시 학습되고 있으니 즉시 중단 |

## 6. 중단과 이어 학습

- `Ctrl+C`로 멈추면 그때까지 저장된 체크포인트(`_000XXXXX.pth`, `_latest.pth`)는 남고, `run_meta.json`에 종료 시각이 기록된다. 큐 스크립트는 남은 작업을 실행하지 않고 멈춘다.
- 같은 run에서 이어 학습 (epoch 번호가 이어진다):

```bash
RUN_NAME=<기존 run 이름> RESUME_FROM=experiments/runs/<기존 run 이름>/nn/<기존 run 이름>_latest.pth \
  bash experiments/scripts/train.sh <변형> <seed> <원래 목표 epoch 수>
```

  `train.log` 앞부분에 `Failed to restore`가 없는지 확인한다. `run_meta.json`의 `segments`에 학습 구간이 쌓인다.

## 7. 평가와 리포트

```bash
bash experiments/scripts/eval_run.sh experiments/runs/<run>     # run 하나 (큐가 자동으로 해 줌)
bash experiments/scripts/eval_all.sh                            # 아직 평가 안 된 모든 체크포인트
bash experiments/scripts/report.sh <T>                          # 표 + 그림 + REPORT.md
```

- 평가는 **학습이 돌지 않을 때** 한다 (GPU 경쟁). 이미 평가한 체크포인트는 건너뛴다 (`FORCE=1`로 다시).
- 평가 1회: 결정적(env 8개) + 확률적(env 32개), 체크포인트당 약 15초. env 생성에 약 30초.
- `report.sh <T>`는 모든 비교를 epoch T 체크포인트로 한다. T 없이 실행하면 각 run의 마지막 체크포인트를 쓰고 T 후보를 제안한다.

`experiments/results/report/`에 생기는 것:

| 파일 | 내용 | 발표 용도 |
|---|---|---|
| `REPORT.md` | 예산 T 후보, baseline 표(seed별 + 평균±표준편차), 변형 비교 표, 첫 실패 원인 분포, H1/H2 판정용 Δ, 자원 표 | 표 1, 표 2, 가설 |
| `fig_learning_curves.png` | 변형별 성공률/진행률/물체 오차 vs epoch (회색 = default) | 그림 1 |
| `fig_train_curves.png` | 학습 중 보상, 에피소드 길이, 진행률 | 보조 |
| `fig_compare_stoch.png`, `fig_compare_det.png` | 변형별 핵심 지표 막대 (점 = seed) | 표 2 시각화 |
| `fig_tradeoff.png` | 잡기 결과 vs 전신 안정성 (가이드 "success vs stability" 곡선) | RQ2 |
| `fig_rq1_sensitivity.png` | 위치/회전 임계값 ×0.5, ×1, ×2에 따른 변화 | RQ1 |
| `fig_series_<run>_e<epoch>.png` | 한 에피소드의 손목 오차(임계값, 창 표시), 물체 오차, 골반 높이, 첫 termination 위치 | 실패 사례 분석 (그림 2) |
| `checkpoints_*.csv`, `episodes_*.csv`, `runs.csv` | 원자료 | 추가 분석 |

## 8. 실패 사례 영상

화면이 있는 상태에서 (viewer 창이 열린다):

```bash
bash experiments/scripts/record_video.sh experiments/runs/<run>/nn/<run>_000XXXXX.pth [이름]
```

- `experiments/results/videos/<이름>.mp4`가 생긴다. 카메라는 골반을 따라간다.
- 실패 사례 고르는 법: `REPORT.md` 6절(첫 실패 원인 분포)과 `episodes_det.csv`의 `first_fail_frame`, `first_cause_*`를 보고 원인별(손목 이탈로 물체를 떨어뜨림 / 잡았지만 궤적 이탈 / 넘어짐·미끄러짐 / 손가락이 못 잡음) 대표를 고른다. 같은 체크포인트의 `fig_series_*.png`와 함께 보여 준다.
- 물리 잡음 때문에 영상 속 에피소드가 표의 env 0과 똑같지 않을 수 있다. 영상과 같은 이름으로 평가 json이 `experiments/results/eval/det_video/`에 저장되니 영상 설명에는 그 숫자를 쓴다.
- viewer에서 **V 키를 누르지 않는다** (시뮬레이션이 멈춤).

## 9. 중간발표 체크리스트 대응

| 중간발표 체크리스트 | 근거 자료 |
|---|---|
| 과제, 입출력 1장 | PROPOSAL_T09.md 1절 |
| baseline 선택 이유와 상태 (Conditional) | PROPOSAL_T09.md 2절, CODE_NOTES.md 1절 |
| 입력 1개 실행 + 측정 (GPU, VRAM, 시간) | `REPORT.md` 9절 자원 표, 이 문서 2절 |
| 같은 조건의 정량 평가 | `REPORT.md` 2~5절 (같은 예산 T, 같은 평가 설정) |
| 관찰한 실패 사례 | `videos/*.mp4` + `fig_series_*.png` + `REPORT.md` 6절 |
| 연구 질문 1개, 반증 가능한 가설 | PROTOCOL.md 6절 (H1/H2), `REPORT.md` 7~8절 |
| 바꿀 모듈, 비교, ablation, 남은 계획 | `configs/variants.yaml`, PROTOCOL.md 2절, 이 문서 4절 |
| 논문 재현과 축소 실험의 차이 | CODE_NOTES.md 1절 |

## 10. 변형 추가와 수정

1. `experiments/configs/variants.yaml`에 변형을 추가한다 (`allowed_keys`에 있는 손목 키만 바꿀 수 있다).
2. `.venv/bin/python experiments/tools/make_configs.py` → `configs/env/<이름>.yaml` 생성. 손목 외 키를 바꾸면 에러가 난다.
3. `bash experiments/scripts/train.sh <이름> 0 <T>`.

`configs/env/*.yaml`을 손으로 고치면 `train.sh`가 시작 전에 거부한다 (`make_configs.py --check`).

## 11. 문제 해결

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: isaacgym` 등 | 저장소 루트에서 실행했는지, `.venv`가 있는지 확인. 재설치는 `isaacgym-env/README.md` |
| `no kernel image is available` | PyPI torch가 깔렸다. `uv sync`로 직접 빌드한 wheel 복구 |
| viewer가 안 뜸 (`record_video.sh`) | `echo $DISPLAY` 확인. SSH라면 데스크톱 세션에서 실행 |
| 평가가 "already evaluated, skipped" | 의도한 것. 다시 하려면 `FORCE=1` |
| `report.sh <T>`가 run을 제외한다고 경고 | 그 run에 epoch T 체크포인트가 없다 (덜 학습됨) |
| `isaacgym-env` 폴더를 옮김 | torch wheel 경로 패치 필요 (`isaacgym-env/README.md` "Moving this directory") |
