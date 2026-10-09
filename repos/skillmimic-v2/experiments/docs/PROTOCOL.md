# SkillMimic-V2 실험 프로토콜 (T09 viability 점검용)

작성: 2026-10-09, **결과를 보기 전에** 정한 규칙이다. 결과를 본 뒤 바꾸면 맨 끝 "변경 기록"에 이유와 함께 적는다.
기준 코드: `Ingrid789/SkillMimic-V2@8e05ad7` + `CODE_NOTES.md` 1절의 로컬 수정.

## 0. 이 실험이 답하려는 질문

T09(WristMimic, 주전자 옮기기)는 RTX 5070 12GB에서 성공률 0%였다 (`../../BASELINE_T09.md`). 원인 후보는 둘이다.

| 가설 | 내용 | 이 저장소에서 보는 것 |
|---|---|---|
| H-hw (하드웨어) | 12GB 때문에 env 수(2048→1024), PhysX 버퍼, 학습량이 줄어서 실패 | 같은 GPU에서 **다른 방법**이 같은 종류의 과제(ParaHome 물체 잡고 들기)를 배우는가 |
| H-task (과제/방법) | 쥐고 드는 순간 자체가 참조 추적 방식에 어렵다 (하드웨어와 무관) | 논문에서 0%였던 SkillMimic(v1)이 여기서도 0%이고 V2는 0보다 큰가 |

SkillMimic-V2 논문(arXiv 2505.02094, Table 2)은 RTX 4090 24GB, 2048 env, clip당 약 1.0B 샘플로 Place-Book, Drink-Cup, Place-Kettle 모두 SR 100%, SkillMimic(v1) baseline은 셋 다 0%를 보고했다.

## 1. 고정하는 것

| 항목 | 값 | 근거 |
|---|---|---|
| Clip | `place_book` (150 frame, s6), `drink_cup` (180, s6), `place_kettle` (100, s10). 저장소에 포함된 그대로 | 논문과 같은 데이터. Place Kettle은 T09와 같은 물체라 다리 역할 |
| 방법 | `ours` = SM + Ours (STF + ATS + HE + buffer node, STG 없음), `sm` = SkillMimic v1, `sm_t` = SM + 시간 (선택) | 논문 5.2절. 코드 대응은 `CODE_NOTES.md` 2절, 정의는 `experiments/tools/methods.py` 한 곳 |
| 환경 설정 | `experiments/configs/env/<clip>.yaml` = 업스트림 `parahome_<clip>_hist60_noisyinit_simpara.yaml` 그대로. 방법 간 차이는 `--asset_file_name`(관측 크기)과 task class뿐 | 모든 방법이 같은 물리(restitution 0, position iteration 10)와 같은 보상 가중치(p 20, r 20, op 1, ig 20 = 논문 표 8) |
| env 수 | **2048** (논문과 같음) | 12GB에서 들어감 (약 10.5GB, `CODE_NOTES.md` 3절) |
| PPO | 업스트림 `parahome.yaml` 그대로 (horizon 32 → 65,536 샘플/update, minibatch 16384, lr 2e-5, σ = e^-2.9). 저장 주기만 250 epoch | 논문 표 8과 같음 |
| 학습 episode 길이 | 60 frame (논문 T = 60) | |
| History encoder | ParaHome 8개 clip으로 3000 epoch, 한 번 학습해서 모든 run에 공유 | 업스트림 `state_prediction_parahome.py`와 같은 설정 |
| 예산 | **3000 epoch = 1.97억 샘플** (논문의 20%) | T09 baseline(1024 env x 32 x 6000 epoch)과 **같은 샘플 수** |
| seed | 0 | 시간 제약. 효과가 경계선이면 seed 추가 (6절) |

## 2. 바꾸는 것 (실험)

| Phase | run | 목적 |
|---|---|---|
| 1 | `ours` x {drink_cup, place_book, place_kettle}, 2048 env, 3000 epoch | 같은 GPU에서 V2가 잡고 들기를 배우는가 (H-hw) |
| 2 | `sm` x {drink_cup, place_kettle}, 같은 예산 | 단순 참조 추적은 여기서도 실패하는가 (H-task, 논문 재현) |
| 3 | `ours` x drink_cup, **1024 env**, 6000 epoch (같은 샘플 수) | env를 T09처럼 절반으로 줄이면 나빠지는가 (H-hw 직접 검증) |

## 3. 평가 규칙 (`experiments/tools/evaluate.py`)

1. 시작 = 참조 frame 2 (업스트림 테스트 명령의 `--state_init 2`), 300 step (10초) 진행, **넘어지면 그 env는 거기서 끝** (재시작 없음).
2. `det` = 결정적 행동 8 env (GPU PhysX가 비트 단위로 재현되지 않으므로 물리 잡음 반복), `stoch` = 학습 때 행동 잡음 32 env.
   `stoch_perturb` = 물체 시작 자세 교란 (z축 ±45°, xy ≤ 10 cm, env마다 seed로 고정) = 논문 ε-NSR.
3. 250 epoch마다 저장된 체크포인트를 모두 `det`로 평가 (학습 곡선). 마지막 체크포인트는 세 방식 모두.
4. 학습 clip = 평가 clip. 일반화를 주장하지 않는다. 시뮬레이터 결과를 실제 로봇 결과라고 하지 않는다.

## 4. 지표 (모두 평가기에 구현, 결과를 보기 전에 고정)

| 지표 | 정의 | 왜 |
|---|---|---|
| **과제 성공** `task_success` | clip의 모든 frame에서 월드 좌표 물체 위치 오차 ≤ 10 cm **그리고** clip 중 넘어지지 않음 | T09의 "과제 성공"과 **같은 정의** → T09 결과와 직접 비교 |
| **들어올림 성공** `lift_success` | 물체가 참조 들어올림 높이의 50% 이상 올라가고 그때 손목 0.2 m 이내 | T09의 병목(쥐고 들기)만 따로 본다 |
| 논문 지표 `paper_success` | 업스트림 metric 그대로: Place는 (골반 z > 0.5, 손목-물체 < 0.2 m, 물체 z > 0.9) > 60 frame, Drink는 (물체 z > 1.2) > 30 frame. 300 step 전체에서 | 논문 SR과 비교 (참고용, `CODE_NOTES.md` 4절의 한계) |
| 과제 진행률, 물체 오차, 몸 오차, 넘어짐, 최대 들어올림, 들고 있던 frame | 실패 위치와 양상 | |

시간 정렬: 시뮬레이션 상태는 그 step의 보상이 비교한 참조 frame(`_curr_ref_obs`)과 비교한다 (업스트림 보상과 같은 정렬, 1 frame 늦음, `CODE_NOTES.md` 4절).

## 5. 판정 기준 (결과 보기 전에 고정)

| 결과 | 해석 |
|---|---|
| Phase 1에서 `ours`의 과제 성공 또는 들어올림 성공 ≥ 0.5 (det, clip 하나 이상) | 12GB/5070은 이 종류의 과제에 **근본적 장애가 아니다**. T09 0%의 주원인은 방법/학습 구조 (H-task) |
| Phase 1 전부 0이고 학습 곡선(보상, 진행률)이 3000 epoch까지 계속 오름 | 예산 부족. 하드웨어 "속도" 문제일 수는 있으나 "메모리" 문제는 아님 (2048 env가 들어갔으므로) |
| Phase 1 전부 0이고 곡선이 평평함 | 이 GPU 설정에서 이 종류의 과제가 막힌다는 증거 (H-hw 쪽). 버퍼/물리 설정 점검 필요 |
| Phase 2 `sm`이 0이고 `ours` > 0 | 논문 재현. 같은 하드웨어에서 **방법**이 결과를 가른다 → T09의 실패도 방법 쪽 설명이 유력 |
| Phase 3 (1024 env)이 Phase 1 (2048)보다 과제 성공 0.25 이상 낮음 | env 수 축소(T09가 강제로 한 것)가 실제로 해롭다 → H-hw 일부 지지 |

차이가 det 8 env의 물리 잡음 범위(±1/8 = 0.125) 안이면 "차이 없음"으로 본다.

## 6. 변경 기록

(아직 없음)
