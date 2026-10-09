# ParaHome 실험 프로토콜 (T09 viability 점검용)

작성: 2026-10-09, **결과를 보기 전에** 정한 규칙이다. 바꾸면 끝의 "변경 기록"에 이유와 함께 적는다.

## 0. 질문

T09 baseline(WristMimic, 주전자 `s110_0_kettle_table2desk`)은 RTX 5070에서 6000 epoch 동안 모든 체크포인트에서 성공 0%였고,
병목은 "쥐고 들어 올리는 순간"(frame 45~60)이었다 (`../../BASELINE_T09.md`).
**그 실패가 주전자 장면 탓인가, 아니면 이 방법/설정에서 쥐고 드는 동작 자체가 막히는가?**

ParaHome는 T09 데이터의 출처다. ParaHome의 다른 물체 장면(컵 마시기, 책 옮기기)에 **T09와 완전히 같은 방법, 설정, 하드웨어, 평가기**를
적용하고 장면만 바꾼다. Push Button은 ParaHome에 없다 (가장 가까운 "가스레인지 손잡이 돌리기"는 관절 물체라 WristMimic 장면이 없다).

## 1. 고정하는 것 (= T09 baseline과 같음)

| 항목 | 값 |
|---|---|
| 코드/환경 | `../t09-wristmimic` (WristMimic + Isaac Gym + 직접 빌드 torch). 이 저장소는 장면만 바꿔 그 스크립트를 부른다 |
| 학습 설정 | `../t09-wristmimic/experiments/configs/env/default.yaml` (12GB PhysX 버퍼), 1024 env, minibatch 16384, 항상 frame 0에서 시작, rollout 300 |
| 평가 | T09 `experiments/tools/evaluate.py` 그대로. 채점 설정 `eval_default.yaml` 고정, 시퀀스 끝까지 reset 없음. det 8 env + stoch 32 env |
| 손 모델 | 각 피험자의 `<subject>_ROM.xml` (데이터 뼈 길이와 0.1 mm 이내로 일치 확인, T09 PROTOCOL 1.1과 같은 규칙) |
| 예산 | 3000 epoch, 500마다 체크포인트 (장면당 약 5.2시간). T09 baseline은 6000 epoch까지 전부 0%였으므로 같은 epoch(500~3000)끼리 비교 |
| seed | 0 |

## 2. 장면 (InterAct 형식으로 `../t09-wristmimic/InterAct/Parahome`에 이미 있음)

| key | 장면 | 요청 Task | 특징 (audit) |
|---|---|---|---|
| `cup_s110` (Phase 1) | `s110_16_cup_drink` (200 frame) | Drink Cup | **T09 주 장면과 같은 사람, 같은 XML**. 오른손, 잡기 frame 23, 들기 frame 47, 0.72 m 들어 올림, 거의 안 걷는다 (0.18 m) |
| `book_s11` (Phase 1) | `s11_book_desk2bookshelf` (150 frame) | Place Book | 두 손, 잡기 16, 들기 45, 0.47 m 들어 올림, 들고 0.94 m 걷는다. 피험자 p2 = SkillMimic-V2의 책 clip과 같은 사람 |
| `cup_s89`, `book_s149` (선택) | 두 번째 사례 | | 왼손 컵 (369 frame) / 책장으로 책 (120 frame) |
| `kettle_s110` (대조군) | `s110_0_kettle_table2desk` | (T09) | T09 baseline 결과를 그대로 가져온다 (다시 학습하지 않음). 오른손, 잡기 26, 들기 47, 들고 1.17 m 걷는다 |

## 3. 지표

T09 평가기의 지표 그대로: 공식 성공(시퀀스 끝까지 실패 조건 없음), 과제 성공(물체 10 cm 이내 + 안 넘어짐), 진행률,
첫 실패 frame과 그 원인(넘어짐/물체/접촉/손목/몸), 물체 오차, 접촉 재현율, 손목 오차.
**첫 실패 frame을 각 장면의 잡기/들기 frame(audit)과 나란히 본다.**

## 4. 판정 기준 (결과 보기 전에 고정)

| 결과 (같은 epoch, det 8 env) | 해석 |
|---|---|
| 컵 또는 책의 공식 성공 ≥ 0.25 (주전자는 0) | T09 실패는 **주전자 장면에 특이적**. 주 장면을 바꾸면 T09는 원래 계획(성공률 비교)으로 살아날 수 있다 |
| 컵/책도 0이고, 첫 실패 frame이 각 장면의 들기 frame ± 15 안에 몰림 | **쥐고 들기가 이 방법/설정의 공통 병목**. 장면 문제가 아님. 하드웨어 탓인지는 SkillMimic-V2 결과(`../skillmimic-v2`)와 같이 판단 |
| 컵/책도 0이고, 실패가 들기 전(잡기 전후)에 남 | 주전자보다 더 이른 단계에서 막힘. 손목/접촉 설정 문제 쪽 |
| 컵/책의 과제 성공 > 0인데 공식 성공 0 | 물체는 따라가지만 손목/접촉 termination이 막는다 → T09의 손목 임계값 연구 질문과 직접 연결 |

차이가 0.125(8 env 중 1개) 이하이면 차이 없음으로 본다.

## 5. 데이터 audit (`experiments/tools/audit_clips.py`, CPU)

원본 ParaHome 207 시퀀스에서 해당 행동의 annotation 구간을 모두 찾아 운동학 지표를 잰다 (손 관절과 물체 표면 거리, 들어올림, 이동,
들고 걷는 거리, 잡기→들기 간격). 스캔 mesh가 닫혀 있지 않아 관통은 재지 않는다. 임계값(손-물체 2.5 cm, 들기 3 cm)은 결과를 보기 전에 정했다.

## 6. 변경 기록

(아직 없음)
