# SkillMimic-V2 노트: 실험 규칙, 코드 변경, 컵 진단

실행 방법과 현재 상태는 `SKILLMIMIC.md`. 이 문서는 **결과를 해석할 때 필요한 것**을 모은다.
경로는 `repos/skillmimic-v2/` 기준. 규칙(1절)은 결과를 보기 전에 정했고, 바꾼 것은 8절 변경 기록에 남긴다.

## 1. 실험 규칙

### 1.1 질문

WristMimic(T09)은 이 GPU에서 0%였다 (`BASELINE_WRISTMIMIC.md`). 원인이 **하드웨어**(12GB라 env 수, 물리 버퍼, 학습량이 줄어서)인지,
**과제/방법**(쥐고 드는 순간이 그 학습 구조에 어려워서)인지 가른다. 같은 GPU에서 다른 방법(SkillMimic-V2)이 같은 종류의 과제를 배우는지 본다.

### 1.2 고정 설정

| 항목 | 값 |
|---|---|
| clip | `drink_cup` (180 frame, ParaHome s6), `place_book` (150, s6), `place_kettle` (100, s10). 저장소에 들어 있는 그대로 |
| 방법 | SM + Ours = STF + ATS + HE + buffer node (STG 없음, 논문 5.3절). 책·주전자 `ours`, 컵 `fix` (= 같은 방법 + 고친 환경, 4절). 정의는 `experiments/tools/methods.py` 한 곳 |
| env 설정 | `experiments/configs/env/<clip>.yaml` = 업스트림 `parahome_<clip>_hist60_noisyinit_simpara.yaml` 그대로. 컵은 `drink_cup_fix.yaml` |
| env 수 | 2048 (논문과 같음) |
| PPO | 논문 표 8 그대로: 65,536 샘플/update, minibatch 16,384, lr 2e-5, γ 0.99, GAE 0.95, clip 0.2, 행동 σ 0.055 고정, MLP 1024-512-512, episode 60 frame |
| 보상 | 논문 표 10: 몸 위치 20, 회전 20, 물체 위치 1, 상대 위치 20, 접촉 항 끔. 곱 형태라 frame당 [0, 1] |
| 예산 | 3000 epoch = 1.97억 샘플 (논문 약 1.0B의 20%). WristMimic baseline과 같은 샘플 수. 논문 예산은 이 GPU에서 clip당 약 45시간 |
| history encoder | ParaHome 8개 clip으로 직접 학습 (미배포), 모든 run 공유 |
| seed | 0 |

### 1.3 평가 (`experiments/tools/evaluate.py`)

1. 참조 frame 2에서 시작, 300 step (10초). **처음 넘어지면 그 env는 끝** (재시작 없음).
2. `det` = 결정적 행동 8 env (GPU 물리가 비트 단위로 재현되지 않아 반복). `stoch` = 학습 때 행동 잡음 32 env.
   `stoch_perturb` = 물체 시작 자세 교란(z축 ±45°, xy ≤ 10 cm) 32 env = 논문 ε-NSR.
3. 250 epoch마다 `det`로 학습 곡선, 마지막 체크포인트는 세 가지 모두.
4. 학습 clip = 평가 clip. 일반화나 실제 로봇 성공을 주장하지 않는다.

| 지표 | 정의 |
|---|---|
| **과제 성공** | clip의 모든 frame에서 물체 오차 ≤ 10 cm **그리고** 넘어지지 않음 (T09와 같은 정의) |
| **들어올림 성공** | 참조 들어올림 높이의 50% 이상 들고 그때 손목 0.2 m 안 |
| 논문 지표 | 업스트림 metric (참고용): Place = 골반 z > 0.5, 손목-물체 < 0.2 m, 물체 z > 0.9가 60 frame 넘게. Drink = 물체 z > 1.2가 30 frame 넘게 |
| 그 밖 | 실패 frame (첫 오차 > 10 cm 또는 넘어짐), 과제 진행률, 물체/몸 오차, 넘어짐, 최대 들어올림, 들고 있던 frame |

### 1.4 판정 (`write_results.py`가 그대로 적용)

| 결과 (최종, det) | 해석 |
|---|---|
| clip 하나라도 과제 성공 또는 들어올림 성공 ≥ 0.5 | 12GB/5070은 근본 장애가 아니다. T09 0%의 원인은 방법 쪽 |
| 모두 < 0.5, 곡선이 오름 | 예산 부족 (메모리 문제는 아님). 같은 run을 더 학습 |
| 모두 < 0.5, 곡선이 평평 | 이 설정에서 막힌다. 물리/환경 점검 |

"오름" = 마지막 1000 epoch 동안 학습 보상 +1% 이상 또는 det 과제 진행률 +0.05 이상. det 8 env의 해상도(0.125) 안의 차이는 "차이 없음".

## 2. 논문/업스트림 대비 다른 점

| 항목 | 업스트림 | 우리 | 결과 영향 |
|---|---|---|---|
| GPU, 학습량 | RTX 4090, clip당 약 1.0B 샘플 | RTX 5070 12GB, 1.97억 (20%) | **있음.** "축소 예산 baseline" (4090의 약 40% 속도) |
| PyTorch | conda torch | 직접 빌드한 torch 2.4.1 cu128 sm_120 (`isaacgym-env/`), uv 환경, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (VRAM 약 0.3 GB 절약) | 없음 |
| ParaHome 학습 명령, 모델, history encoder | 공개 안 됨 (README에 ParaHome 절이 비어 있음) | 코드를 읽고 맞춤 (3절). encoder는 업스트림 학습 코드로 직접 학습 (3000 epoch, 6분) | encoder가 원본과 다를 수 있음 |
| 이어 학습 | `--resume_from`이 가중치와 입력 정규화만 복원 → epoch 0부터 다시, optimizer 초기화, 앞 체크포인트 **덮어씀** | epoch, optimizer, 정규화, frame, ATS 상태, 난수까지 복원. 임시 파일에 쓰고 이름 바꾸기 | 없음 |
| 체크포인트 | 50 epoch, 최신만 | 10 epoch, 모두 유지 | 없음 |
| 평가 | player 성공률 (넘어져 재시작한 뒤의 frame까지 합산) | 우리 평가기 (1.3절) | 측정 방식만 |
| 속도 | 장면 pickle을 env마다 다시 읽고, reset마다 참조 history를 다시 계산 | 둘 다 캐시 (같은 값) | 없음 (시작 105초, reset 빨라짐) |
| **컵 환경** | 4절의 버그 | method `fix` | **있음.** 두 run을 나란히 보고 |

## 3. 논문 방법 ↔ 코드

| 논문 표 2 | task class | asset (관측 크기를 파일 **이름**이 정함) | 플래그 |
|---|---|---|---|
| SM | `SkillMimicParahome` | `mocap_parahome_boxhand.xml` (obs 1022) | 없음 |
| SM + T | `SkillMimicParahomePhase` | `..._refobj.xml` (+6 위상) | 없음 |
| **SM + Ours** (`ours`) | `SkillMimicParahomeLocalHistRISBuffernode` | `..._hist.xml` (+3 history) | `--reweight --reweight_alpha 1.0` (ATS), `--state_init_random_prob 0.1` (STF), `--enable_buffernode`, `--hist_length 60 --history_embedding_size 3 --hist_ckpt ...` (HE) |
| SM + Ours, 고친 환경 (`fix`) | `SkillMimicParahomeLocalHistRISBuffernodePhase` | `..._histphase.xml` (+3 +6) | `ours`와 같음, env `drink_cup_fix.yaml` |

- 업스트림 책 yaml만 asset이 `_refobj.xml`이라 그대로 SM+Ours를 돌리면 `shape mismatch [N,1025] vs [N,1028]`. 그래서 방법마다 asset을 명시한다.
- 코드의 ATS 가중치는 exp(−5·r̄) (논문 표 9는 λ=10). 코드 값을 쓴다.

## 4. 컵 진단과 수정 (2026-10-11)

첫 컵 run(업스트림 환경, 3000 epoch)은 0%였고 손이 컵 22 cm 안으로 거의 못 들어갔다. 정책 없이 시뮬레이션을 직접 점검했다
(`experiments/probe/`, GPU 몇 분씩). 고친 것은 모두 method `fix`와 `drink_cup_fix.yaml`로만 켜지고 책·주전자에는 영향이 없다.

| # | 문제 (업스트림) | 증거 | 수정 (`fix`) |
|---|---|---|---|
| 1 | **식탁 충돌 형상이 거칠다** (자동 분해, voxel 약 6 cm) | 정책 없이 컵을 참조 위치에 두면 1 step에 +5.1 cm 튕기고 +4.2 cm에 놓인다. 따로 떨어뜨리면 21 cm 미끄러진다. 첫 run에서 손이 닿기 전에 컵이 14~19 cm 움직인 것과 같다 | 식탁을 메시 윗면에 맞춘 **상자 하나**로 (1.207 × 0.79 m, yaw 165.5°). 컵이 참조에서 **0.3 cm** 안에 정지. 세밀한 자동 분해는 2048 env에서 GPU 메모리 부족 |
| 2 | **컵 1.49 kg, 손잡이 고리가 메워짐** (밀도 2500, 10 hull) | 실제 머그 약 0.35 kg. 시연은 손잡이를 잡는다 | 손으로 나눈 볼록 조각 (몸통 + D자 손잡이 막대 4개, 고리는 비움, `experiments/tools/make_cup_collision.py` → `cup_fix.urdf`). **0.35 kg** |
| 3 | **history encoder 입력 배치 버그** | encoder 학습과 reset은 참조 몸 **71개**(61~70번은 0) 배치, 실행 중에는 시뮬레이터 몸 **61개** → 181~211번 차원에 회전값. 같은 상태에서 그 구간 차이 0.61 (값 크기 0.45) | 같은 배치로 맞춤 (`histLayoutFix`), reset 때 낡은 상태 대신 참조 행. 차이 0.07 |
| 4 | **위상 정보가 사실상 없다** | 관측에 참조 목표도 시간도 없고 3차원 history embedding뿐. history가 시뮬레이터 frame으로 차면 위상 오차 평균 약 20 frame (무작위 약 40, 3번을 고쳐도 같다). 논문 표 2도 시간 정보 없는 SM은 컵 0%, 위상을 넣은 SM+T는 99.9% | 위상 관측 6차원 (SM+T와 같은 식). HE는 그대로 |
| 5 | **컵을 놓쳐도 episode가 계속된다** (넘어질 때만 끝) | 컵을 쓰러뜨린 뒤에도 몸 따라 하기 보상을 받는다 | 학습 중 컵이 참조에서 0.25 m 넘게 벗어나면 끝 (`objTerminationDist`). 평가는 그대로 |
| 6 | **(데이터, 못 고침) 시연 손 자세로는 컵이 안 붙잡힌다** | 컵을 든 참조 상태(frame 40/60/90)에서 몸을 고정하고 손을 참조 자세로 유지해도 바로 떨어진다. 손가락을 더 오므려도(6가지), 마찰 1~8, 겹침 허용, 침투 해소 속도, 손잡이 두께를 바꿔도 같다. 손가락이 손잡이 고리 근처에 있지만 약 1 cm 어긋나 있다 | 없음. 정책이 시연과 다른 자신의 잡기를 찾아야 한다 |

**검증.**
- 2048 env에서 관측 1031, VRAM 10.7 GB, 약 5,400 samples/s (3000 epoch 약 10시간). 끊고 다시 실행하면 ATS 상태까지 이어진다. 평가 3가지 동작.
- 400 epoch 비교 (`experiments/probe/results/`, 컵 형상만 중간판): 업스트림은 손이 닿기 전에 컵을 19~33 cm 밀어냈다. 고친 환경은 컵이 0.2 cm 안에 그대로 있었고 몸 오차는 6~8 cm였다 (업스트림 9~36 cm). frame당 학습 보상은 0.43으로 업스트림 0.20의 2배다. 하지만 손을 컵으로 뻗지는 않았고 (손목-컵 25~27 cm, 참조 14 cm), 컵을 든 상태에서 시작해도 4~8 frame 안에 놓쳤다 (6번).
- 고친 환경의 평가에서 넘어짐이 많다 (frame 71~81). 학습이 컵을 놓치면 끝나서 그 뒤 frame을 거의 겪지 않기 때문이다.

**기대.** 1~5는 학습을 원천적으로 막던 문제라 고쳤다. 남은 벽은 6번이다. 3000 epoch 안에 정책이 잡기를 찾을지는 학습해 봐야 안다.
250 epoch마다의 평가에서 들어올림 성공, 최대 들어올림, 들고 있던 frame이 먼저 움직일 것이다. 3000 epoch 뒤에도 들기가 없으면,
다음 수단은 시연 손 자세를 컵 형상에 맞게 고치는 것이다.

점검 도구 (학습 없음, 각각 몇 분):

```bash
.venv/bin/python experiments/probe/scene_probe.py drink_cup                      # 사람 없이 식탁+컵: 지지면, 안착, 질량
.venv/bin/python experiments/probe/env_probe.py --clip drink_cup --method fix    # 참조 재생, 정지 상태, history 입력
.venv/bin/python experiments/probe/env_probe.py --clip drink_cup --method fix --only grip60   # 손에 든 상태에서 버티는가 (그 밖 hits, gap, phase)
```

## 5. 업스트림 코드의 함정 (해석에 필요)

| # | 내용 | 우리 처리 |
|---|---|---|
| 1 | 학습 종료는 넘어짐(골반 z < 0.25 m)뿐. WristMimic은 물체/손목/접촉 이탈로도 끝낸다 | 성공은 평가기에서 따로 판정. `fix`는 물체 이탈도 종료 |
| 2 | 학습 episode는 60 frame이고 매번 참조의 랜덤 frame에서 시작 (ATS가 낮은 보상 frame을 자주 뽑음) | 구조적 차이로 기록 |
| 3 | 업스트림 player 성공률은 넘어져 재시작한 뒤의 frame까지 합산 | 우리 평가기는 첫 넘어짐까지만 |
| 4 | 주전자 참조 자체가 논문 지표의 "60 frame 넘게"를 45 frame만 만족 | 논문 지표는 참고용, 주 지표는 과제 성공 |
| 5 | 보상은 k번째 step 후 상태를 참조 frame (시작 + k − 1)과 비교 (1 frame 늦음) | 평가도 같은 정렬 |
| 6 | STF의 자세 거리 식이 참조를 제곱한다 (`norm(noisy - motion**2)`, 버그로 보임) | 방법의 일부라 그대로 |
| 7 | 물체가 참조보다 높게 놓인다: 책 약 3.8 cm, 컵 4.2 cm (가구 충돌 형상) | 책은 그대로(10 cm 기준의 여유가 줄어듦), 컵은 고침 |
| 8 | history 입력 배치 버그, 위상 정보 없음 (4절 3·4번) | `ours`는 그대로(논문 코드), `fix`만 고침 |

## 6. 측정값 (2048 env)

| run | 최대 VRAM | 속도 | 3000 epoch |
|---|---|---|---|
| 컵 `ours` | 11.4 GB (11.0에서 접촉이 늘며 커짐, epoch 624에 GPU 할당 실패로 한 번 죽음) | 6,070 samples/s | 9.1시간 |
| 책 `ours` | 10.0 GB | 6,600 | 8.3시간 |
| 주전자 `ours` | 9.8 GB (60 epoch) | 5,800 | 약 10시간 |
| 컵 `fix` | 10.7 GB (30 epoch) | 5,400~5,700 | 약 10시간 |

- 메모리 대부분은 PhysX GPU 버퍼 (torch는 약 2.5 GB). 12GB에서 여유가 1~2 GB라 학습 중 GPU를 쓰는 다른 프로그램은 끈다.
- 병목은 물리 (step당 PhysX 0.15 s, Python 0.07 s). 평가는 체크포인트당 약 11초 + 환경 생성 40초. 체크포인트 하나 43 MB.

## 7. 바뀐 파일 (업스트림 대비, 모두 `[3dv]` 주석)

| 파일 | 내용 |
|---|---|
| `skillmimic/learning/{skillmimic_agent,common_agent}.py`, `skillmimic/run.py` | 전체 상태 이어 학습, 원자적 저장, 난수/ATS 상태 저장, run 이름 고정(`SMV2_RUN_NAME`) |
| `skillmimic/env/tasks/skillmimic_parahome.py` | ATS 상태 저장/복원 |
| `skillmimic/env/tasks/humanoid_object_task.py` | 장면 pickle 캐시. env cfg 선택 옵션 `staticBoxes`, `staticVhacd`, `dynamicUrdf`, `dynamicVhacd`, `dynamicRestOffset` (없으면 업스트림 그대로) |
| `skillmimic/env/tasks/skillmimic_parahome_localhist.py` | 참조 history 캐시, `histLayoutFix`, 새 task `...BuffernodePhase` (위상 + 물체 조기 종료) |
| `skillmimic/env/tasks/humanoid_task.py`, `mjcf/mocap_parahome_boxhand_histphase.xml`, `skillmimic/utils/parse_task.py` | `fix`의 관측 크기, task 등록 |
| `skillmimic/data/assets/obj/cup_fix.urdf`, `cup_fix/` | 컵 충돌 형상 (`make_cup_collision.py`가 만듦) |
| `skillmimic/data/motions/ParaHome/s6, s10, s22` | `experiments/data/parahome_seq/`로의 symlink (가구 위치를 읽음, `setup.sh`) |
| `experiments/configs/` | clip별 env yaml (업스트림 복사본), `drink_cup_fix.yaml`, `train/parahome.yaml` (업스트림 + 10 epoch 저장, 전체 상태 이어 학습) |
| `experiments/tools/` | `methods.py` (방법 정의 한 곳), `evaluate.py`, `write_results.py`, `ckpt.py` (이어 학습할 체크포인트), `train_hist_encoder.py`, `make_cup_collision.py` |
| `experiments/scripts/` | `train.sh`, `eval_run.sh`, `run_baseline.sh` (`run_*.sh`가 부름), `setup.sh`, `pipeline_check.sh`, `_common.sh` (clip별 method) |
| `experiments/probe/` | 정책 없는 환경 점검 도구, 진단 때의 평가 결과 |

## 8. 변경 기록

| 날짜 | 변경 | 이유 | 결과를 본 뒤인가 |
|---|---|---|---|
| 2026-10-10 | 범위를 SkillMimic-V2 세 run으로 (SM v1 비교, 1024 env 비교는 뺌) | 팀이 baseline을 SkillMimic-V2로 정함 | 아니오 |
| 2026-10-10 | 체크포인트 10 epoch마다 모두 유지, 이어 학습을 전체 상태 복원으로 | 업스트림 이어 학습이 epoch와 optimizer를 초기화하고 체크포인트를 덮어씀 | 아니오 |
| 2026-10-10 | 판정의 "곡선이 오름"을 숫자로 (보상 +1% 또는 진행률 +0.05) | 자동 판정 | 아니오 |
| 2026-10-11 | 컵을 고친 환경(`fix`)에서 새로 학습. 업스트림 환경 run은 비교용으로 유지. 평가와 판정 규칙은 그대로 | 4절 | **예** (첫 컵 run 0%를 본 뒤). 그래서 두 run을 나란히 보고 |
| 2026-10-11 | 학습이 죽으면 자동으로 2번 이어 하기, `write_results.sh`는 새 번호 문서로 | 첫 컵 run의 GPU 할당 실패, 기존 결과 보존 | 운영 변경 |
