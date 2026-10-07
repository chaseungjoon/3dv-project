# T10 CrossFormer 실험 실행 가이드

과제 **T10 Cross-embodiment robot manipulation**의 baseline 평가 가이드다.
baseline은 가이드가 지정한 **CrossFormer** (공식 130M 체크포인트 `hf://rail-berkeley/crossformer`, inference only)이다.

- 바로 실행: [QUICKSTART.md](QUICKSTART.md)
- 규칙, 지표 정의, 판정 기준: [docs/PROTOCOL.md](docs/PROTOCOL.md)
- 코드에서 확인한 사실, 원본 대비 변경, 함정: [docs/CODE_NOTES.md](docs/CODE_NOTES.md)

> 모든 명령은 저장소 루트 `~/code/3dv-project/repos/t10-crossformer` 에서 실행한다.

---

## 0. 무엇을 하는가

T10의 정의: 여러 로봇 데이터로 학습한 정책을 **고정(frozen)** 한 채, 로봇 사이의 관측/action 표현 차이(좌표계, 단위, 제어 주기)를
보정해 전이를 좋게 한다. 이 폴더는 그 **출발점(baseline)** 을 잰다: "공식 사용법 그대로 쓰면 두 로봇에서 어떻게 되는가,
그리고 변환을 틀리면 얼마나 나빠지는가".

가이드의 순서 ("offline inference를 먼저 확인하고, 호환되는 simulator wrapper가 있을 때만 closed-loop")를 그대로 따른다.

| Phase | 내용 | 산출물 |
|---|---|---|
| 0 | 공식 Colab 예제(`inference_pretrained.ipynb`) 로컬 재현 | `results/notebook/` |
| 1 | **Baseline**: held-out episode에서 공식 사용법 2가지(언어 조건, goal image 조건)의 오프라인 action 오차, 실행 제약 위반 | REPORT.md 1절 |
| 2 | 조건(task)과 history 길이의 영향 | REPORT.md 2절 |
| 3 | 규약 불일치 probe: 관측(crop, 좌우 반전), 제어 주기, action 단위(통계) | REPORT.md 3절 |
| 4 | **Closed-loop 성공률**: SimplerEnv visual matching (WidowX 4 task, Google Robot 3~4 task) | REPORT.md 4절 |
| 5 | 시뮬레이터 검증: 같은 설치에서 공식 Octo-Base가 논문 값을 재현하는가 | REPORT.md "Phase 5" |
| 6 | Viability screening: 출력 변환(언어, ensemble, action_scale)만 바꾼 closed-loop, baseline과 같은 episode끼리 비교 | REPORT.md "Phase 6" |

### 두 embodiment

| | WidowX 250 | Google Robot |
|---|---|---|
| 데이터 | `bridge_dataset` 1.0.0 (rail 사본 = CrossFormer가 학습한 그 버전) | `fractal20220817_data` 0.1.0 (RT-1) |
| 학습 mixture 가중치 | 0.17 | 0.17 |
| held-out 평가 subset | `val` split 앞 4 shard, 215 episode | `train[95%:]` 중 마지막 4 shard, 342 episode |
| 제어 주기 | 5 Hz | 3 Hz |
| action (표준화 후) | EEF delta xyz (m) + euler delta + gripper(1=열림) | world_vector + rotation_delta (RT-1 단위) + gripper(1=열림) |
| closed-loop | SimplerEnv `widowx_*` 4 task | SimplerEnv `google_robot_*` (coke can, move near, drawer) |

세 번째로 `taco_play` (Franka, 15 Hz, `test` split 46 episode)도 받아 두었다 (오프라인 전용, 선택).
action 규약의 실측값(단위, 좌표계 회전, 지연)은 `results/conventions/CONVENTIONS.md`.

---

## 1. 폴더 구조

```
experiments/
├── QUICKSTART.md              실행 순서만
├── README.md                  이 문서
├── docs/
│   ├── PROTOCOL.md            규칙, 지표 정의, 판정 기준, 가설
│   └── CODE_NOTES.md          코드 분석, 원본 대비 변경, RTX 5070 메모, 함정
├── configs/
│   ├── datasets.yaml          평가 데이터 (URL, shard 선택, held-out 근거, 제어 주기)
│   ├── variants.yaml          오프라인 variant (task 조건, history, frame skip, 이미지 처리)
│   └── sim_variants.yaml      closed-loop 변환 variant (ensemble, scale, 회전 변환, 통계)
├── tools/
│   ├── download_subset.py     shard 몇 개만 받아 split "val"로 노출 + held-out 검사
│   ├── data.py                학습과 같은 전처리로 episode 로드 (standardize, 카메라, Lanczos 224)
│   ├── conventions.py         로봇별 action→EEF 이동 선형 fit (단위, 좌표계, 지연, gripper 부호)
│   ├── evaluate.py            오프라인 추론 → runs/<variant>/<dataset>/pred.npz (정규화 상태로 저장)
│   ├── metrics.py             지표 + 제약 위반 + 통계 probe → results/metrics/
│   ├── sim_policy.py          SimplerEnv용 CrossFormer 정책 (공식 Octo wrapper와 같은 변환)
│   ├── sim_suites.py          SimplerEnv 공식 visual-matching episode grid
│   ├── sim_eval.py            closed-loop 평가 (--policy crossformer|octo-base) → results/sim/ (단계 지표 포함)
│   ├── sim_backfill_stages.py 단계 지표가 없는 예전 closed-loop 결과에 영상 이름에서 단계 지표 추가
│   ├── notebook_repro.py      Phase 0
│   └── aggregate.py           표, 그림, REPORT.md
├── scripts/                   실행 스크립트 (setup, pipeline_check, phase0~4, report)
├── results/                   (git 포함) conventions/, metrics/, sim/, notebook/, report/
├── data/                      (git 제외) held-out subset, 약 1.6 GB
├── checkpoints/               (git 제외) crossformer/ 500 MB, tfhub/ 600 MB (문장 인코더)
├── third_party/               (git 제외) ManiSkill2_real2sim (SimplerEnv scene asset), octo (Octo 1.0, 대조 정책)
└── runs/                      (git 제외) pred.npz, closed-loop 영상과 action log
```

## 2. 각 단계 상세

### Phase 1: baseline

`evaluate.py`는 episode의 모든 시점 t에 대해 학습과 같은 입력을 만든다: t에서 끝나는 5-frame history
(episode 시작 전은 첫 frame 반복 + `timestep_pad_mask=False`), `image_primary` 하나, 224×224 Lanczos3.
`single_arm` head가 4-step action chunk를 내고, **정규화된 출력 그대로** 저장한다. unnormalize는 `metrics.py`에서 하므로
"다른 로봇 통계로 unnormalize" 같은 단위 probe에 추가 추론이 필요 없다.

- `baseline_lang`: 공식 notebook의 언어 조건 (goal image 0 + Universal Sentence Encoder embedding)
- `baseline_goal`: 학습과 같은 조건 (episode 마지막 frame을 goal image로, 언어 embedding 0). 왜 이것이 "학습과 같은" 조건인지는 CODE_NOTES 3절.

### Phase 2, 3

Phase 2는 같은 가중치로 조건 방식(`cond_*`)과 history(`hist_w1`, `hist_w2`)를 바꾼다. Phase 3는 규약을 일부러 틀린다:
`obs_center_crop` (Google Robot 카메라 320×256을 정사각형으로 자름. Bridge는 이미 정사각형이라 변화 없음),
`obs_hflip` (카메라 좌우 반전), `rate_x2`/`rate_x3` (k frame마다 관측, GT는 k step 합 = 제어 주기를 1/k로).
action 단위 probe는 추가 추론 없이 REPORT의 "action-unit probe" 표에 나온다.

### Phase 4: closed-loop

SimplerEnv의 공식 평가기(`maniskill2_evaluator`)와 공식 스크립트의 episode grid를 그대로 쓰고, 정책만 CrossFormer로 바꿨다.
변환 계층은 SimplerEnv의 Octo wrapper와 같다 (통계 unnormalize → chunk ensemble → euler→axis-angle → gripper 규칙).
CrossFormer는 Octo 코드베이스 위에 만들어졌고 같은 Bridge/fractal 데이터로 학습해서 이 변환이 공식 규약에 가장 가깝다.
goal image는 closed-loop에서 알 수 없으므로 언어 조건(`sim_baseline`)과 조건 없음(`sim_notask`)만 있다.

| suite | 로봇 | task | episode | max step |
|---|---|---|---|---|
| bridge | WidowX | stack cube, carrot on plate, spoon on towel, eggplant in basket | 4 × 24 | 60 (eggplant 120) |
| coke_can | Google Robot | pick coke can (가로/세로/세움) | 3 × 25 위치 × 4 URDF = 300 | 80 |
| move_near | Google Robot | move near | 60 × 4 URDF = 240 | 80 |
| drawer (선택) | Google Robot | open/close top/middle/bottom drawer | 6 × 9 자세 × 4 URDF = 216 | 113 |

## 3. 사전 점검 (2026-10-07 완료)

| 항목 | 결과 |
|---|---|
| jax 0.6.2 GPU (RTX 5070, sm_120) | OK. 원본 pin jax 0.4.20은 sm_120 kernel이 없어 실행 불가 |
| 체크포인트 로드, 공식 notebook step 1/2 재현 | OK (`results/notebook/`) |
| held-out subset 다운로드와 held-out 검사 (`SUBSET.json`) | OK |
| 오프라인 추론 속도 | 약 10 ms/window (batch 32). batch 128은 12 GB에서 OOM |
| GPU 메모리 | 오프라인 최대 약 5.5 GB, closed-loop 약 4.5 GB (장치 전체, 데스크톱 약 0.7 GB 포함) |
| 오프라인 재현성 | 같은 입력 두 번: 최대 차이 1.7e-6 (사실상 결정적, L1 head는 sampling 없음) |
| SimplerEnv (SAPIEN 2.2.2, Vulkan) 렌더링, ray tracing | OK, headless |
| closed-loop 재현성 | 같은 episode 두 번: 첫 step 차이 1e-6 수준, 60 step 중 끝에서 gripper 1번 다름 (아주 드물게 결과가 바뀔 수 있음) |
| `pipeline_check.sh` 전체 | 통과, 약 8분 |

## 4. 데이터나 variant 추가

- 데이터셋: `configs/datasets.yaml`에 항목 추가 (CrossFormer `HEAD_TO_DATASET["single_arm"]`에 있어야 하고, 체크포인트
  `dataset_statistics.json`에 이름이 있어야 함) → `.venv/bin/python experiments/tools/download_subset.py <name>` →
  `conventions.py <name>` → `DATASETS="... <name>"`.
- 오프라인 variant: `configs/variants.yaml`에 한 줄 추가 → `bash -c 'source experiments/scripts/_common.sh; run_offline <variant>'`
- closed-loop variant: `configs/sim_variants.yaml`에 한 줄 → `VARIANTS="<variant>" bash experiments/scripts/phase4_closed_loop.sh`
- shard 수를 늘리려면 `datasets.yaml`의 `shards`를 바꾸고 setup을 다시 실행 (예: `first:8`).
