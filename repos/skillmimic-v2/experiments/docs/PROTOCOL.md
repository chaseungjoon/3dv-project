# SkillMimic-V2 baseline 프로토콜 (컵 / 책 / 주전자)

작성: 2026-10-09 (viability 점검용), 2026-10-10 baseline 전용으로 정리 (6절). **결과를 보기 전에** 정한 규칙이다. 결과를 본 뒤 바꾸면 맨 끝 "변경 기록"에 이유와 함께 적는다.
기준 코드: `Ingrid789/SkillMimic-V2@8e05ad7` + `CODE_NOTES.md` 1절의 로컬 수정. 계획: `../../../../PROPOSAL_T09.md`.

## 0. 이 실험이 답하려는 질문

T09(WristMimic, 주전자 옮기기)는 RTX 5070 12GB에서 성공률 0%였다 (`../../BASELINE_T09.md`). 원인 후보는 둘이다.

| 가설 | 내용 | 이 baseline에서 보는 것 |
|---|---|---|
| H-hw (하드웨어) | 12GB 때문에 env 수(2048→1024), PhysX 버퍼, 학습량이 줄어서 실패 | 같은 GPU에서 **다른 방법**이 같은 종류의 과제(ParaHome 물체 잡고 들기)를 배우는가 |
| H-task (과제/방법) | 쥐고 드는 순간 자체가 WristMimic의 학습 구조(엄격한 종료, frame 0 시작)에 어렵다 (하드웨어와 무관) | SkillMimic-V2가 같은 GPU에서 0보다 큰 성공을 내는가 |

SkillMimic-V2 논문(arXiv 2505.02094, Table 2)은 RTX 4090 24GB, 2048 env, clip당 약 1.0B 샘플로 Place-Book, Drink-Cup, Place-Kettle 모두 SR 100%, SkillMimic(v1) baseline은 셋 다 0%를 보고했다.

## 1. 고정하는 것

| 항목 | 값 | 근거 |
|---|---|---|
| Clip | `place_book` (150 frame, s6), `drink_cup` (180, s6), `place_kettle` (100, s10). 저장소에 포함된 그대로 | 논문과 같은 데이터. Place Kettle은 T09와 같은 물체라 다리 역할 |
| 방법 | `ours` = SkillMimic-V2 = SM + Ours (STF + ATS + HE + buffer node, STG 없음) | 논문 5.3절. 코드 대응은 `CODE_NOTES.md` 2절, 정의는 `experiments/tools/methods.py` 한 곳 |
| 환경 설정 | `experiments/configs/env/<clip>.yaml` = 업스트림 `parahome_<clip>_hist60_noisyinit_simpara.yaml` 그대로. asset은 `--asset_file_name`으로 명시 (관측 크기) | 세 clip이 같은 물리(restitution 0, position iteration 10)와 같은 보상 가중치(p 20, r 20, op 1, ig 20 = 논문 표 10) |
| env 수 | **2048** (논문과 같음) | 12GB에서 들어감 (약 10.5GB, `CODE_NOTES.md` 3절) |
| PPO | 업스트림 `parahome.yaml` 그대로 (horizon 32 → 65,536 샘플/update, minibatch 16384, lr 2e-5, σ = e^-2.9). 저장 주기만 10 epoch (번호 붙여 모두 유지) | 논문 표 8과 같음 |
| 학습 episode 길이 | 60 frame (논문 T = 60) | |
| History encoder | ParaHome 8개 clip으로 3000 epoch, 한 번 학습해서 모든 run에 공유 | 업스트림 `state_prediction_parahome.py`와 같은 설정 |
| 예산 | **3000 epoch = 1.97억 샘플** (논문의 20%) | T09 baseline(1024 env x 32 x 6000 epoch)과 **같은 샘플 수** |
| seed | 0 | 시간 제약. 효과가 경계선이면 seed 추가 (6절) |

## 2. 실험 (baseline)

| 스크립트 | run | 목적 |
|---|---|---|
| `run_cup.sh` | `drink_cup_ours_n2048_s0`, 3000 epoch | 같은 GPU에서 SkillMimic-V2가 잡고 들기를 배우는가 (한 손, 들어서 입으로) |
| `run_book.sh` | `place_book_ours_n2048_s0`, 3000 epoch | 같음 (두 손으로 들어 옮기기) |
| `run_kettle.sh` | `place_kettle_ours_n2048_s0`, 3000 epoch | 같음. T09(WristMimic)가 0%였던 물체 |

각 스크립트는 학습 → 평가(3절)까지 하고 결과 문서는 쓰지 않는다. `write_results.sh`가 `../../BASELINE_SKILLMIMIC.md`를 만든다.
끊긴 run은 같은 명령으로 마지막 체크포인트부터 이어진다 (epoch, optimizer, 정규화, ATS 상태, 난수 상태 복원. `CODE_NOTES.md` 1.2절).
이어 붙인 run은 끊기지 않은 run과 같은 학습으로 본다. 다만 GPU PhysX 비결정성 때문에 비트 단위로 같지는 않다 (끊기지 않은 run도 마찬가지).

## 3. 평가 규칙 (`experiments/tools/evaluate.py`)

1. 시작 = 참조 frame 2 (업스트림 테스트 명령의 `--state_init 2`), 300 step (10초) 진행, **넘어지면 그 env는 거기서 끝** (재시작 없음).
2. `det` = 결정적 행동 8 env (GPU PhysX가 비트 단위로 재현되지 않으므로 물리 잡음 반복), `stoch` = 학습 때 행동 잡음 32 env.
   `stoch_perturb` = 물체 시작 자세 교란 (z축 ±45°, xy ≤ 10 cm, env마다 seed로 고정) = 논문 ε-NSR.
3. 250 epoch마다의 체크포인트를 `det`로 평가 (학습 곡선). 마지막 체크포인트(3000)는 세 방식 모두.
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

| 결과 (최종 체크포인트, det) | 해석 |
|---|---|
| clip 하나 이상에서 과제 성공 또는 들어올림 성공 ≥ 0.5 | 12GB/5070은 이 종류의 과제에 **근본적 장애가 아니다**. T09 0%의 주원인은 방법/학습 구조 (H-task) |
| 세 clip 모두 < 0.5이고 학습 곡선이 마지막 1000 epoch 동안 계속 오름 | 예산 부족. 하드웨어 "속도" 문제일 수는 있으나 "메모리" 문제는 아님 (2048 env가 들어갔으므로). 같은 run을 이어서 더 학습 |
| 세 clip 모두 < 0.5이고 곡선이 평평함 | 이 GPU 설정에서 이 종류의 과제가 막힌다는 증거 (H-hw 쪽). 버퍼/물리 설정 점검 필요 |

- "계속 오름" = 마지막 1000 epoch 동안 학습 보상(앞뒤 각 250 epoch 평균)이 1% 이상 늘거나 det 과제 진행률이 0.05 이상 늘어남. 둘 다 아니면 "평평".
- 차이가 det 8 env의 물리 잡음 범위(±1/8 = 0.125) 안이면 "차이 없음"으로 본다.
- `write_results.py`가 이 규칙을 그대로 적용해 `BASELINE_SKILLMIMIC.md` 2절에 적는다. 세 run이 끝나기 전에는 "잠정"으로 표시한다.

## 6. 변경 기록

| 날짜 | 변경 | 이유 | 결과를 본 뒤인가 |
|---|---|---|---|
| 2026-10-10 | 범위를 SkillMimic-V2 baseline 세 run으로 줄임. SM(v1) baseline(구 Phase 2)과 1024 env 비교(구 Phase 3)는 뺌 | 팀이 T09 baseline을 SkillMimic-V2로 정함 (`PROPOSAL_T09.md`) | 아니오 (본 실험 전) |
| 2026-10-10 | 체크포인트 250 → 10 epoch마다, 모두 유지. 학습 곡선 평가는 그대로 250 epoch마다 | 끊겨도 잃는 학습을 최대 10 epoch로. 평가 비용은 그대로 | 아니오 |
| 2026-10-10 | 이어 학습을 전체 상태 복원으로 고침 (`CODE_NOTES.md` 1.2절) | 원본 `--resume_from`은 epoch 0부터 다시 세고 optimizer를 초기화했다 (2026-10-09 점검 로그에서 확인) | 아니오 |
| 2026-10-10 | 5절 "곡선이 계속 오름"을 숫자로 정함 (보상 +1% 또는 진행률 +0.05, 마지막 1000 epoch) | 자동 판정에 필요 | 아니오 |
