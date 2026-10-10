# SkillMimic-V2 코드 노트 (2026-10-09, 2026-10-10 이어 학습 수정 추가)

업스트림: `github.com/Ingrid789/SkillMimic-V2@8e05ad7` (2025-07-24). README에는 BallPlay/Locomotion 명령만 있고
**"Household Manipulation Policy" 절이 비어 있다** (ParaHome 모델도 미공개, TODO에 남아 있음). 그래서 ParaHome 학습 경로는
코드를 읽고 맞췄다. 아래는 그 결과와 원본 대비 바뀐 것.

## 1. 원본 대비 바뀐 것

### 1.1 실행 환경 (동작에 영향 없음)

| 항목 | 원본 | 우리 | 이유 |
|---|---|---|---|
| GPU | RTX 4090 24GB | RTX 5070 12GB (sm_120) | |
| PyTorch | conda torch (SkillMimic v1 설치법) | 직접 빌드한 torch 2.4.1 cu128 sm_120 (`../../isaacgym-env/wheels`) | PyPI cp38 torch에 sm_120 커널 없음 (T09와 같은 휠) |
| 환경 관리 | conda | `pyproject.toml` + uv (numpy 1.23.5, rl-games 1.1.4, pytorch-lightning 1.9.0, six) | T09와 같은 방식 |
| `PYTORCH_CUDA_ALLOC_CONF` | 없음 | `expandable_segments:True` (`_common.sh`) | 최대 VRAM 11.1 → 10.8 GB. 할당기만 바뀌고 계산은 같다 |

### 1.2 코드 수정

| 파일 | 수정 | 동작 영향 |
|---|---|---|
| `skillmimic/env/tasks/humanoid_object_task.py` | ParaHome 장면 pickle(`object_transformations.pkl`, 약 40 MB)을 env마다 다시 읽던 것을 경로별 1회로 캐시 | 없음 (읽기 전용). 2048 env 시작이 약 105초 빨라짐 |
| `skillmimic/run.py` | 환경변수 `SMV2_RUN_NAME`이 있으면 run 폴더 이름에 타임스탬프를 붙이지 않음 (`full_experiment_name`) | 없음. `train.sh`의 자동 이어 학습용 |
| `skillmimic/learning/skillmimic_agent.py` | config `resume_full_state: True`이면 `--resume_from`이 `restore_full_state()`를 부른다: 가중치, 입력/value/amp 정규화, epoch, optimizer, frame, task 학습 상태(`env_state`), 난수 상태(torch/cuda/numpy/python) 복원. 원본 `restore()`는 가중치와 입력 정규화만 읽어서, 이어 학습이 **epoch 0부터 다시 세고** optimizer와 value 정규화가 초기화되고 **앞 구간의 번호 체크포인트를 덮어썼다** (2026-10-09 점검: 20 epoch 뒤 이어서 30까지 → 실제로 31 epoch를 더 돌고 `_00000010`, `_00000020`을 새로 씀) | 이어 학습이 끊기지 않은 학습과 같아짐. 처음부터 하는 학습에는 영향 없음 |
| `skillmimic/learning/common_agent.py` | `save()`: 임시 파일(`.pth.tmp`)에 쓰고 `os.replace` (끊겨도 깨진 체크포인트 없음). `get_full_state_weights()`: 난수 상태 추가. `train()`: 이어 학습이면 frame 카운터를 0으로 되돌리지 않고, 최고 보상 체크포인트 관리(상위 2개 유지)를 기존 파일에서 이어감 | 없음 |
| `skillmimic/run.py` (`RLGPUEnv`) | `get_env_state`/`set_env_state`가 task의 `get_train_state`/`set_train_state`를 부른다 (rl_games가 체크포인트에 `env_state`로 넣음. 원본은 항상 `None`) | 없음 |
| `skillmimic/env/tasks/skillmimic_parahome.py` | `get_train_state`/`set_train_state`: ATS 상태(`progress_buf_total`, frame별 보상 `motion_time_seqreward`, `time_sample_rate`, clip 가중치) | 없음 (이어 학습 때 ATS가 처음부터 다시 쌓이지 않음) |
| `skillmimic/data/motions/ParaHome/s6, s10, s22` | `experiments/data/parahome_seq/`(git 제외, 87 MB)의 원본 ParaHome 시퀀스로의 symlink (`setup.sh`, 없으면 seq.zip을 받아 세 장면만 남김). 2026-10-10까지는 `../parahome/data/seq`를 가리켰다 | 업스트림 코드가 책상/식탁 위치를 `ParaHome/s10/object_transformations.pkl` frame 0에서 읽는데 저장소에 없다. 가구 위치는 s6과 s10에서 같다 (확인) |

### 1.3 우리가 추가한 설정 (업스트림 파일은 그대로 둠)

| 파일 | 내용 |
|---|---|
| `experiments/configs/env/<clip>.yaml` | 업스트림 `parahome_sm/parahome_<clip>_hist60_noisyinit_simpara.yaml`의 복사본. 세 clip 모두 같은 물리/보상. `assetFileName`은 방법마다 `--asset_file_name`으로 덮어쓴다 |
| `experiments/configs/train/parahome.yaml` | 업스트림 `parahome.yaml` + `save_frequency 50→10`, `save_intermediate: True` (10 epoch마다 번호 붙은 체크포인트, 지우지 않음), `resume_full_state: True`. 학습에는 영향 없음 |
| `experiments/tools/ckpt.py` | 이어 학습할 체크포인트 고르기: `nn/`의 모든 체크포인트 중 epoch가 가장 크고 정상적으로 읽히는 것. 남은 `.pth.tmp` 삭제 |
| `experiments/tools/train_hist_encoder.py` | 업스트림 `utils/state_prediction_parahome.py`는 import하면 하드코딩된 경로로 학습을 시작하고 체크포인트도 저장하지 않는다. 같은 dataset/model 클래스를 그대로 쓰고 CLI와 저장만 붙였다 |

## 2. 논문 방법 ↔ 코드 대응 (README에 없음, 코드에서 확인)

baseline은 **SM + Ours**만 쓴다 (`experiments/tools/methods.py`). SM, SM + T 행은 논문 표 2를 읽을 때 참고용.

| 논문 (Table 2) | task class | asset (관측 크기를 정함) | 학습 플래그 |
|---|---|---|---|
| SM | `SkillMimicParahome` | `mocap_parahome_boxhand.xml` (obs 1022) | 없음 (RSI만) |
| SM + T | `SkillMimicParahomePhase` | `mocap_parahome_boxhand_refobj.xml` (+6 = 위상 t/len을 6번 반복) | 없음 |
| **SM + Ours** | `SkillMimicParahomeLocalHistRISBuffernode` | `mocap_parahome_boxhand_hist.xml` (+3 = history embedding) | `--reweight --reweight_alpha 1.0` (ATS), `--state_init_random_prob 0.1` (STF: 잡음 섞인 초기 상태 + 가장 비슷한 참조 frame으로 재표본), `--enable_buffernode` (buffer node), `--hist_length 60 --history_embedding_size 3 --hist_ckpt ...` (HE) |

- 논문 5.2절: ParaHome에서는 STG를 쓰지 않는다 (clip이 서로 다른 물체라 연결이 무의미) → `--graph_file`, `--state_switch_prob` 없음.
- **관측 크기는 XML 내용이 아니라 파일 이름으로 정해진다** (`humanoid_task.py:_setup_character_props`). `_refobj`, `_hist`, 기본 XML은 내용이 완전히 같다.
- 업스트림 `parahome_place_book_hist60_noisyinit_simpara.yaml`만 asset이 `_refobj.xml`이다 (SM+T용으로 보임). 그대로 SM+Ours를 돌리면 `shape mismatch [N,1025] vs [N,1028]`로 죽는다. 우리는 방법별로 asset을 명시한다.
- ParaHome용 history encoder는 배포되지 않았다 (`hist_encoder/`에는 BallPlay, Locomotion뿐). ParaHome 8개 clip으로 3000 epoch 학습 (약 6분, 최종 MSE 0.00023).

## 3. 측정값 (2026-10-09, pipeline_check + clip별 6~15 epoch probe, 2048 env)

| clip | 방법 | 최대 GPU 메모리 (데스크톱 약 0.5 GB 포함) | 학습 속도 | s/epoch | 3000 epoch |
|---|---|---|---|---|---|
| drink_cup | SM + Ours | 10.8 GB | 6,100 samples/s | 10.7 | 약 8.9시간 |
| drink_cup | SM | 10.9 GB (expandable segments 없이) | 10,100 | 6.5 | 약 5.4시간 |
| place_book | SM + Ours | 9.5 GB | 6,200 | 10.6 | 약 8.8시간 |
| place_book | SM | 9.2 GB | 10,500 | 6.2 | 약 5.2시간 |
| place_kettle | SM + Ours | 10.1 GB | 5,500 | 12.0 | 약 10시간 |
| place_kettle | SM | **11.2 GB** (15 epoch 동안 일정, 증가 없음) | 9,000 | 7.3 | 약 6.1시간 |

- torch 몫은 할당 2.3 GB / 예약 2.6 GB뿐이고 나머지(약 7.5 GB)는 PhysX GPU 버퍼 + CUDA context다. 메모리는 시작 직후 정해지고 학습 중 늘지 않았다.
- 12GB 카드에서 여유는 1~2.5 GB다. 학습 중에는 GPU를 쓰는 다른 프로그램(브라우저 하드웨어 가속 등)을 끈다.
- 평가 (det 8 env, 300 step): 체크포인트당 약 11초 + 환경 생성 약 40초.
- 병목은 물리다 (프로파일: step당 PhysX 0.15 s, Python 루프 약 0.07 s). 2048 env에서 1024보다 샘플 처리량이 거의 늘지 않는다.
- 논문 BallPlay 1.3B 샘플/24시간(4090) = 약 15,000 samples/s → 이 GPU는 약 2.4배 느리다. 논문 1.0B 샘플은 clip당 약 45시간이라
  예산을 T09와 같은 샘플 수(1.97억)로 잡았다 (PROTOCOL.md 1절).
- 체크포인트 하나 43 MB. 10 epoch마다 모두 유지하면 3000 epoch run당 약 300개, 13 GB. 저장은 약 1초라 학습 속도에 영향 없음.
- T09(WristMimic)는 2048 env에서 OOM이었지만 SkillMimic-V2는 2048 env가 들어간다 (손가락이 상자 형태, PhysX 버퍼 배수 10).

## 4. 원본 코드의 함정 (평가 해석에 중요)

| # | 내용 | 우리 처리 |
|---|---|---|
| 1 | **학습 중 종료 조건은 넘어짐(골반 z < 0.25 m)뿐**이다. 물체/손목/접촉이 참조에서 벗어나도 episode가 끝나지 않는다. T09(WristMimic)는 물체, 손목, 접촉 이탈로 끝낸다 | 구조적 차이로 기록. 그래서 성공 판정은 우리 평가기에서 따로 한다 |
| 2 | 학습 episode는 60 frame이고 **매번 참조의 랜덤 frame에서 시작**한다 (RSI + 보상 기반 재가중). T09는 항상 frame 0에서 300 frame | T09 진단 run에서 랜덤 시작이 "들고 걷기"를 살렸던 것과 같은 축. 보고서에서 비교 |
| 3 | 업스트림 player의 성공률(metric)은 env가 중간에 넘어져 재시작해도 이전 생의 frame을 합산한다 (`metric_manager.reset(done_indices)`가 마지막에 끝난 env만 reset) | 우리 평가기는 한 생(첫 넘어짐까지)만 센다 |
| 4 | Place 지표는 "> 60 frame"이 필요한데 **Place-Kettle 참조 자체가 100 frame 중 45 frame만** 조건을 만족한다 (Place-Book 80, Drink-Cup 61/30 필요) | 논문 SR은 clip보다 긴 episode로 잰 것으로 보인다. 우리는 300 step을 굴리고 논문 지표는 참고로만 쓴다. 주 지표는 T09와 같은 과제 성공 |
| 5 | 보상은 k번째 step 후의 상태를 참조 frame (시작 + k − 1)과 비교한다 (1 frame 늦음) | 평가도 같은 정렬(`_curr_ref_obs`)을 쓴다. 30 Hz에서 물체가 1 m/s로 움직이면 약 3 cm |
| 6 | `noisyinit_find_most_similarity_state`의 자세 거리 `norm(noisy - motion**2)`은 참조를 제곱한다 (버그로 보임) | STF 동작의 일부라 **고치지 않았다** (논문 결과도 이 코드로 나왔다고 본다) |
| 7 | 시뮬레이션에서 책은 참조보다 약 3.8 cm 높게 놓인다 (VHACD 충돌 형상 위에 얹힘) | 10 cm 과제 기준의 여유가 그만큼 줄어든다. 보고서에 명시 |
| 8 | 학습 episode가 끝나는 frame은 clip 끝(`envid2episode_lengths`)이고, 테스트에서 `--episode_length`를 주면 그 길이로 끝난다 (clip 뒤는 참조가 0으로 채워짐) | 평가는 clip 안의 frame만 추적 지표에 쓴다 |
