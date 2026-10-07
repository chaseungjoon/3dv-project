# T10 코드 노트: CrossFormer에서 확인한 사실과 원본 대비 변경

발표 슬라이드와 연구 단계 설계용. 줄 번호는 `repos/t10-crossformer` 기준.
원본: `github.com/rail-berkeley/crossformer@4a56b64` (top-level git에 수정 없는 snapshot이 따로 commit되어 있어 `git diff`로 변경을 볼 수 있다).

---

## 1. 원본 대비 로컬 수정

| 파일 | 수정 | 이유 |
|---|---|---|
| `pyproject.toml` | uv 환경 추가 (원본의 black/isort 설정은 유지) | 원본은 conda + `requirements.txt` (jax 0.4.20, TF 2.15) |
| `crossformer/utils/typing.py` | `jax.random.KeyArray` → `jax.Array` | jax 0.6에서 제거됨 |
| `crossformer/model/*.py`, `crossformer/utils/*.py`, `scripts/finetune.py`, `scripts/server.py` | `jax.tree_map(` → `jax.tree.map(`, `jax.tree_leaves(` → `jax.tree.leaves(` | jax 0.6에서 제거됨 |

모델 코드의 동작은 바꾸지 않았다. 나머지는 모두 `experiments/` 안의 새 파일이다.

### 1.1 환경 (RTX 5070, sm_120)

| 패키지 | 버전 | 메모 |
|---|---|---|
| python | 3.10 | 원본 README와 같음 |
| jax[cuda12] / flax / orbax | 0.6.2 / 0.10.7 / 0.11.16 | 원본 pin jax 0.4.20 (cuda11)은 sm_120 kernel이 없어 실행 불가. 2026-09-29 설치 점검에서 0.6.2 + flax 0.10.7 조합 확인 |
| tensorflow | 2.19.0 | **CPU 전용으로 씀** (`tf.config.set_visible_devices([], "GPU")`). TF는 sm_120 kernel이 없어 GPU를 쓰면 PTX JIT 경고가 뜨고 jax와 메모리를 다툰다 |
| tensorflow-metadata | 1.16.1 | 1.17+는 protobuf 6으로 생성되어 TF 2.19(protobuf 5 이하)와 충돌 |
| setuptools | < 80 | `tensorflow_hub`가 `pkg_resources`를 import |
| dlimp | kvablack/dlimp@5edaa46 + `override-dependencies = tensorflow==2.19.0` | dlimp가 `tensorflow==2.15.0`을 pin |
| opencv-python | (headless 아님) | `mani_skill2_real2sim`이 요구 |
| simpler-env / mani-skill2-real2sim | SimplerEnv@eddb569 / ManiSkill2_real2sim@ef7a4d4 (git) | SAPIEN 2.2.2. scene asset은 `experiments/third_party/`에 (setup.sh) |

`crossformer` 패키지는 설치하지 않고 `PYTHONPATH=<repo>`로 쓴다 (`_common.sh`). 원본 `setup.py`와 pyproject의 project 이름 충돌을 피하려고.

## 2. 체크포인트에서 확인한 사실

- 파라미터 129.75M, `300000/` 단일 step. window 5. 문장 인코더는 Universal Sentence Encoder large v5 (TF Hub, 512차원, CPU에서 실행).
- head 4개: `bimanual` (14차원 × 100 step), `single_arm` (7 × 4), `nav` (2 × 4), `quadruped` (12 × 1). 모두 `L1ActionHead`, `clip_pred=False` → **gripper 출력에 clip이 없어 [0,1] 밖이 나올 수 있다**.
- L1 head는 sampling이 없다 (`rng`는 받지만 쓰지 않음) → 같은 입력이면 같은 출력.
- `dataset_statistics.json`: 35개 데이터셋. single_arm 데이터는 모두 `mask = [T,T,T,T,T,T,F]` → **gripper 차원은 정규화되지 않는다** (head가 0~1 값을 직접 냄).
- single-arm 데이터셋은 관측으로 `image_primary` 하나만 쓴다. 손목 카메라 key는 config에서 주석 처리(`None`), proprio도 `bimanual`/`quadruped`만 있다 (`scripts/configs/pretrain_config.py`, 체크포인트 `config.json`의 `dataset_kwargs_list`).
- 실제 학습 mixture (`config.json` `sample_weights`): bridge_dataset 0.17, fractal 0.17, omnimimic_gnm 0.17, aloha_pen_uncap 0.085, go1 0.085, droid_wipe 0.085, ... taco_play 0.010. 코드의 `oxe_dataset_mixes.py`의 `CROSS_EMBODIMENT`와 다르다 (fractal은 코드 mix에는 없고 체크포인트 mix에는 있다). **체크포인트의 `config.json`을 기준으로 삼아야 한다.**

## 3. 언어 조건에 관한 중요한 사실 (Phase 1에 variant 두 개를 두는 이유)

체크포인트 `config.json`의 `traj_transform_kwargs`:
`goal_relabeling_strategy="uniform"`, `max_goal_distance=15`, `task_augment_strategy="delete_task_conditioning"`, **`keep_image_prob=1.0`**.

`crossformer/data/utils/task_augmentation.py:84` `delete_task_conditioning`은 goal image가 있으면 확률 `keep_image_prob`로 image를 남기고
**언어를 padding(0)으로 지운다**. goal relabeling이 항상 goal image를 만들므로, 이 설정으로는 언어가 학습 중 항상 지워진다.
즉 공개 체크포인트는 (적어도 마지막 학습 단계에서는) **goal-image 조건으로 학습**되었다. 그런데:

- 공식 notebook은 마지막에 `task = model.create_tasks(texts=[...])` (언어 조건)으로 추론한다.
- `create_tasks(texts=...)`는 goal image를 0으로 채우고, `create_tasks(goals=...)`는 언어 자리에 `USE("")`(0이 아닌 벡터)를 넣는다 (`crossformer_model.py:76`).
- 언어 embedding은 pad mask와 상관없이 FiLM 입력으로 그대로 들어간다 (`tokenizers.py` `ImageTokenizer`, `task_film_keys`). goal image는 관측과 채널로 쌓인다(early fusion).

그래서 "학습과 같은" goal 조건은 goal image + **언어 0**이다 (`baseline_goal`, `evaluate.py` `TaskBuilder`). notebook 방식 goal 조건은 `cond_goal_nb`.
개발 중 30 episode 관찰: Google Robot에서 언어 조건 ≈ 조건 없음 (norm L1 0.567 vs 0.560), goal 조건 0.488, oracle sub-goal 0.415.
**closed-loop에서는 goal image를 알 수 없으므로 언어 조건밖에 없다**. 이 차이가 closed-loop 성공률에 그대로 나타날 수 있다 (PROTOCOL H6).

## 4. 학습 입력 형식 (평가가 따라야 하는 것)

- history: `crossformer/data/traj_transforms.py:12` `chunk_act_obs`. episode 시작 전은 첫 frame을 반복하고 `timestep_pad_mask=False`.
  action chunk는 `[t, t+H)`, episode 끝 너머는 마지막 action 반복 (우리 평가는 이 부분을 지표에서 제외).
- 이미지: `dlimp.transforms.resize_image` = `tf.image.resize(..., "lanczos3", antialias=True)` 후 반올림. **종횡비를 유지하지 않는다** (fractal 320×256도 224×224로 늘임).
- Bridge action은 `relabel_actions` (`data_utils.py:359`)로 **상태 차분으로 다시 계산**된다 (다음 상태 − 현재 상태). 그래서 Bridge의 action→EEF fit은 R² = 1.0, 단위 = m, 지연 0이 정확히 나온다.

## 5. 로봇별 action 규약 (실측, `results/conventions/CONVENTIONS.md`)

| | WidowX (Bridge) | Google Robot (fractal) | Franka (TACO, 선택) |
|---|---|---|---|
| 제어 주기 | 5 Hz | 3 Hz | 15 Hz |
| action → EEF 이동 | 1.0 m/unit, 회전 0°, 지연 0, R² 1.00 | 약 0.19~0.29 m/unit (축마다 다름), 회전 약 3°, **지연 1 step**, R² 0.77 | 약 0.012~0.017 m/unit, 회전 약 6°, **지연 2 step**, R² 0.75 |
| step 이동 중앙값 / p99 | 13 mm / 53 mm | 16 mm / 82 mm | 5 mm / 22 mm |
| gripper | action 1 = 열림, 다음 상태와 corr +0.83 | action 1 = 열림, `gripper_closed` 상태와 corr −0.71 (상태 부호 반대) | action 1 = 열림 |
| action std (x) | 0.0096 | 0.069 | 0.23 |

같은 정책의 같은 head가 낸 "x 방향 0.01"이 로봇에 따라 1 cm, 약 3 mm, 약 0.15 mm가 된다. 통계를 바꿔 끼우면 크기가 5배 틀린다
(Phase 3 단위 probe). 이것이 T10이 말하는 단위/좌표계/주기 차이의 실제 크기다.

## 6. SimplerEnv 연결

- `tools/sim_policy.py`는 SimplerEnv의 `simpler_env/policies/octo/octo_model.py` 변환을 그대로 옮겼다:
  통계 unnormalize → 4-step chunk의 겹치는 예측 균등 평균 (`ActionEnsembler`, temperature 0) → translation × scale,
  euler(roll, pitch, yaw) → axis-angle → gripper (WidowX: 0.5에서 ±1, Google Robot: 이전−현재의 상대 명령을 0.5 넘으면 15 step 유지).
- 차이: Octo wrapper는 history가 덜 찼을 때 짧은 window를 넣지만, 우리는 학습과 같이 첫 frame 반복 + pad mask False로 5 frame을 채운다.
- 평가는 SimplerEnv의 `maniskill2_evaluator`를 그대로 호출한다. 성공 판정, episode 길이, 초기 자세, 배경 overlay 모두 공식 값 (`tools/sim_suites.py`가 공식 `scripts/*.sh`의 인자를 옮긴 것).
- 점검 episode 관찰: WidowX는 목표 쪽(접시)으로 바로 가는 등 지시를 무시하는 듯한 동작, Google Robot은 몇 step 만에 gripper를 닫고 이동이 매우 작음 (raw 약 0.01 = 약 3 mm/step, 데이터 중앙값 16 mm).

## 7. 성능과 메모리 (RTX 5070 12 GB, Ryzen 9 9950X, 64 GB)

| 항목 | 값 |
|---|---|
| 프로세스 시작 (TF/USE 로드 + 체크포인트 + jax 컴파일) | 약 30~50초 (첫 실행은 USE 600 MB 다운로드로 약 2분) |
| 오프라인 추론 | 약 7.5~10 ms/window (batch 32, 이미지 디코딩 포함 wall 기준 약 10 ms) |
| batch 128 | OOM (jax가 2.5 GB 한 번에 할당 시도) |
| GPU 최대 사용 (장치 전체) | 오프라인 약 5.5 GB, closed-loop 약 4.5 GB. `XLA_PYTHON_CLIENT_PREALLOCATE=false` 필수 (없으면 jax가 75% 선점 → 9.7 GB로 보임, SAPIEN과 충돌 가능) |
| SimplerEnv step | 약 20~30 ms (렌더링 포함), episode당 약 6~10초 (Bridge 60 step, Google 80 step) |

## 8. 함정

- `pkg_resources` 없음, protobuf gencode 충돌, dlimp의 TF pin: 1.1 표.
- zsh에서 `for f in $files` 는 단어 분리가 안 된다 (스크립트는 모두 bash).
- 원본 `.gitignore`가 `*.png`를 무시한다 → `experiments/.gitignore`에서 `!*.png`로 리포트 그림을 다시 포함.
- SimplerEnv는 `DISPLAY=""`로 headless 렌더링한다. `GLFW error: X11: Failed to open display`는 무해하다.
- `get_args()`는 `sys.argv`를 파싱하고, 공식 스크립트는 argparse 접두어 축약(`--robot-init-x` → `--robot-init-x-range`)에 기대고 있다. `sim_eval.py`가 같은 방식으로 호출한다.
