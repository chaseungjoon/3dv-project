# WristMimic 코드 분석 노트 (T09)

`repos/t09-wristmimic`을 읽고 확인한 내용. 발표의 "원본 대비 바뀐 것" 슬라이드와 결과 해석에 쓴다.
코드 위치는 저장소 루트 기준이다.

## 1. 원본 대비 바뀐 것 (발표 때 반드시 명시)

### 1.1 실행 환경 때문에 바꾼 것 (동작에 영향 있음)

| 항목 | 논문/원본 | 우리 | 이유 |
|---|---|---|---|
| GPU | RTX 3090 24GB | RTX 5070 12GB (sm_120) | |
| PyTorch | 1.x / cu116 | 2.4.1 직접 빌드 (CUDA 12.8, sm_120) | PyPI cp38 torch에 sm_120 커널 없음 (`isaacgym-env/README.md`) |
| numpy | 1.21.1 | 1.23.5 | 직접 빌드한 torch의 C-API |
| 병렬 환경 | 2048 | **1024** | 2048은 CUDA OOM |
| epoch당 샘플 | 2048×32 = 65,536 (minibatch 4개) | 1024×32 = 32,768 (minibatch 2개) | 위와 같음. epoch당 gradient step 절반 (mini_epochs 6 × 2 = 12) |
| PhysX 버퍼 | `default_buffer_size_multiplier` 20, `max_gpu_contact_pairs` 34.6M | 5, 8.4M (`parahome_train_12gb.yaml`) | 원래 값은 메모리 할당 실패 |
| `silu` 활성화 | rl-games 1.1.4에 없음 (원본 버그) | network builder에 등록 | `ValueError: silu` |
| 학습 예산 | 수만 epoch (스크립트에 `exp225_00043500` 흔적) | T epoch (PROTOCOL.md 5절) | 시간 |
| 손 모델 | README 예시 `s110_intermimic_ROM.xml` | `s110_ROM.xml` | 데이터 뼈 길이와 정확히 일치 (PROTOCOL.md 1.1). 원본 학습 스크립트와 같은 파일 |
| 평가 | `parahome_test.yaml` | `experiments/configs/env/eval_default.yaml` | 원본 테스트 설정은 이 코드로 실행 불가 (2.1) |

### 1.2 기록만 추가한 것 (동작 영향 없음)

| 파일 | 추가 내용 |
|---|---|
| `intermimic/env/tasks/intermimic_multi_obj.py` | `extras`에 프레임별 termination 원인(`termination_causes`, `kinematic_causes`)과 손목 진단값(`wrist_diag`) 저장. `INTERMIMIC_IMAGE_DIR`로 viewer 이미지 저장 위치 지정 |
| `intermimic/learning/common_agent.py` | wandb로만 가던 지표(진행률, 보상 성분, 시간)를 TensorBoard `wm/*`에도 기록 |
| `intermimic/run.py` | `INTERMIMIC_EXACT_RUN_NAME=1`이면 run 폴더 이름에 타임스탬프를 붙이지 않음 (train.sh가 직접 붙임) |
| `intermimic/env/tasks/intermimic_multi_obj.py` | 설정 키 `startFrame`(기본 0): `stateInit: Start`가 시작하는 참조 프레임. 기본값이면 원본과 같다. 진단 평가(`evaluate.py --start_frame`)에서만 쓴다 |

검증: 기록 코드는 텐서를 읽고 복사만 한다. 파이프라인 점검에서 원본 `run.py --test` 결과가 우리 평가 분포 안에 있음을 확인했다. 물리 잡음 때문에 비트 단위 비교는 불가능하다 (PROTOCOL.md 4.3).

## 2. 원본 코드의 함정

### 2.1 원본 테스트 설정이 그대로는 안 돈다
`parahome_test.yaml`은 `numObsNew: 3492`인데, 관측 코드(`_compute_observations_new`)가 실제로 만드는 크기는 **2236**이다.
(현재 980 = 고유감각 778 + 접촉 31 + 물체 15 + IG 156, 참조 628 × 2 = 프레임 t+1, t+8.)
또 PhysX 버퍼가 24GB용이고 보상 가중치도 학습 설정과 다르다. 그래서 평가는 학습 설정(기본 손목 값)과 같은 `eval_default.yaml`로 한다. 보상 가중치는 termination에 영향이 없으므로 채점에 무관하다.

### 2.2 원본 테스트 루프는 끝나지 않는다
`intermimic_players.py`의 `num_test_episodes = 10000` 하드코딩 + `max_steps = 27000`. 한 에피소드(155프레임) 뒤에도 계속 돈다. 우리 평가기(`experiments/tools/evaluate.py`)는 에피소드 1개(시퀀스 끝)에서 멈추고, env 하나를 만들어 여러 체크포인트를 순서대로 평가한다.

### 2.3 첫 스텝의 관측은 0이다
reset은 상태만 바꾸고 관측(`obs_buf`)을 다시 계산하지 않는다. 새 프로세스의 첫 행동은 0 관측에서 나온다 (학습 중에는 이전 에피소드의 마지막 관측). 평가기는 체크포인트마다 `obs_buf`를 0으로 만들어 원본 테스트와 같은 조건을 재현하고, 앞서 평가한 체크포인트의 영향을 없앤다.

### 2.4 설정 파일에 있지만 쓰이지 않는 값

| 키 | 실제 |
|---|---|
| `terminationHeight: 0.15` | 무시됨. 넘어짐 판정은 `_build_termination_heights`에 하드코딩된 **골반 높이 0.3 m** |
| `resetWindowBefore/After` | 출력만 됨. 손목 reset 창은 grasp 창과 같다 (3절) |
| `fingerWindowBefore/After`, `gfp*`, `gfr*` | 쓰이지 않음. 손가락 자세 보상 함수(`compute_grasp_reward_new`)는 호출되지 않는다 (호출하면 없는 키 `finger_start` 때문에 에러) |
| `gp`, `gho`, `gsn` (테스트 설정) | 쓰이지 않음 |
| `ig` reset (`compute_ig_reward`의 `reset_ig`) | 계산만 되고 termination에 들어가지 않음 |
| `physicalBufferSize: 3` (PSI) | `_compute_reset`의 PSI 블록은 시퀀스 길이 < `rolloutLength`이면 바로 return → 이 장면(155 < 300)에서는 꺼진 것과 같다. `rolloutLength`를 줄이면 켜지는데 원본 코드가 상태 332차원을 참조 358차원에 넣으려다 `RuntimeError`로 죽는다 (2026-10-05 진단 run에서 발생). 진단 설정은 1로 끈다 |
| CLI `--reset_transition_frame` | `resetTransitionFrame`(쓰이지 않는 키)을 바꾼다. 실제 키는 `resetTransitionFrame1/2` |

### 2.5 학습은 앞 300프레임만 본다
(주의) `stateInit: Hybrid`/`Random`의 시작 프레임 범위는 `[0, 시퀀스 길이 − rolloutLength)`다. `rolloutLength: 300`이면 155프레임 장면에서 `max(1, 155−300) = 1` → **항상 프레임 0**이 되어 Start와 같다. 랜덤 시작을 쓰려면 `rolloutLength`를 시퀀스보다 짧게 해야 한다 (`diag_hybrid.sh`는 60).

`rolloutLength: 300`, `stateInit: Start`이므로 학습 에피소드는 항상 프레임 0에서 시작해 최대 300프레임이다. 주 장면은 155프레임이라 전부 덮는다. 더 긴 장면을 쓰면 뒷부분은 학습되지 않는다.

### 2.6 물체 오차의 좌표계
원본 테스트가 출력하는 `obj_pos_err`는 **골반 heading 기준 좌표**의 오차다 (물체 높이는 월드 z). 몸이 쓰러지거나 돌면 물체가 제자리여도 오차가 커진다. 우리는 월드 좌표 오차를 주 지표로, 원본 값은 `*_local_mean`으로 같이 기록한다.

## 3. 손목 항목이 실제로 작동하는 방식

### 3.1 창
한 손의 참조 첫 접촉 프레임을 c라 하면 (손목, 손가락 포함 팔 body의 참조 접촉 > 0.1이 처음 나오는 프레임):

```
창 전체   [c − graspWindowBefore, c + graspWindowAfter]          기본 [c−10, c+15]
stage 1   [c − graspWindowBefore, c + resetTransitionFrame1]     기본 [c−10, c−2]
stage 2   (c + resetTransitionFrame1, c + resetTransitionFrame2]  기본 (c−2, c+12]   "key part"
stage 3   (c + resetTransitionFrame2, c + graspWindowAfter]      기본 (c+12, c+15]
```

주 장면은 오른손 c = 29 → **프레임 19~44 (0.87초)** 에서만 손목 항목이 작동한다. 나머지 프레임에서 손목은 손목 전용 항 없이 몸 보상으로만 추적된다. 왼손은 접촉이 없어 창이 없다.

### 3.2 손목 reset 조건 (`_compute_wrist_reset_condition`)
stage k에서 `위치 오차 > wristPosResetThreshold_k` 또는 `회전 오차 > wristRotResetThreshold_k`이고, 프레임 ≥ 10이면 참. 기본 임계값: stage 1, 3은 0.15 m / 0.5 rad, stage 2는 0.07 m / 0.2 rad.
손목 reset은 몸(key body 최대 오차 > 0.35 m), 물체(물체 점 평균 오차 > 0.5 m)와 OR로 묶여 kinematic termination이 된다. 접촉 termination은 참조에서 손이 물체에 닿아 있는데 시뮬레이션 손이 안 닿는 상태가 10프레임 넘게 지속될 때다.

### 3.3 손목 보상 (`_compute_wrist_error_and_reward`)
`exp(−‖Δp‖² · gwp_k)`와 `exp(−Δθ · gwr_k)`, 창 밖에서는 가중치 0이라 보상 1. 몸 보상 `rb = rp · 손목위치 · 손목회전 · rr · energy`에 곱해진다. 전체 보상은 `rb · ro(물체) · rig(IG) · rcg(접촉)`.

### 3.4 "손목 제약 없음"의 한계
`wrist_off`에서도 손목 위치는 몸 위치 보상 `rp`의 key body 목록에 들어 있다 (창 안: `KeyBodies_No_Elbow_No_Shoulder` + 접촉 없는 팔, 가중치 `gbp`=10; 창 밖: 모든 key body, 가중치 `p`=30). 즉 `wrist_off`는 "손목 **전용** 항 제거"이지 손목 추적 완전 제거가 아니다. 발표에서 이렇게 표현한다.

## 4. 데이터

`InterAct/Parahome/<장면>/<...>.pt` 한 파일에 `data`(프레임 × 721), `objects`, `joints`가 있다. 파일 이름의 `kettle,desk,diningtable_111`은 등장 물체와, 보상 대상 물체 인덱스(첫 '1'의 위치 = kettle)를 뜻한다.

| 장면 | 프레임 | 접촉 손 | 첫 접촉 c |
|---|---|---|---|
| `s110_0_kettle_table2desk` (주) | 155 | 오른손 | 29 |
| `s110_16_cup_drink` (예비, 같은 XML) | 200 | 오른손 | 23 |
| `s108_30_kettle_desk2table` | 180 | 오른손 | 18 |
| `s79_0_cup_desk2table` | 180 | 왼손 | 50 |
| `s99_0_cup_table2desk` | 180 | 왼손 | 0 |
