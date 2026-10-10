# T09 제안서: SkillMimic-V2 baseline (컵 / 책 / 주전자)

과제: **T09. Wrist-guided humanoid manipulation** (Theme 5, Conditional candidate)
Baseline: **SkillMimic-V2** (Yu et al., SIGGRAPH 2025, arXiv 2505.02094), 코드 `Ingrid789/SkillMimic-V2@8e05ad7`
장면: ParaHome **Drink Cup**, **Place Book**, **Place Kettle** (clip 하나에 정책 하나)
환경: RTX 5070 12GB, Ubuntu 24.04, Isaac Gym Preview 4, 직접 빌드한 torch 2.4.1 (sm_120)
작성: 2026-10-10 · 상태: **준비와 점검 완료, baseline 학습 전**

> **결정 (2026-10-10).** 팀은 T09의 baseline을 WristMimic에서 **SkillMimic-V2**로 바꾸고, `BASELINE_T09_VARIETY.md`에서 고른
> 세 clip(컵, 책, 주전자)으로 baseline을 만든다. 이유는 셋이다.
> (1) WristMimic은 이 GPU에서 주전자를 쥐고 드는 순간(frame 45~60)을 넘지 못해 성공률 0%였다 (`BASELINE_T09.md`).
> (2) SkillMimic-V2는 WristMimic 논문의 비교 대상이고(과제 안내서 T09 항목), 같은 ParaHome 데이터와 같은 Isaac Gym을 쓰며,
> 논문에서 세 clip 모두 성공률 100%다. (3) 논문과 같은 2048 env가 12GB에 들어간다 (WristMimic은 2048 env에서 메모리 부족).
> 실행 방법: `repos/skillmimic-v2/QUICKSTART.md`. 결과: `BASELINE_SKILLMIMIC.md` (`write_results.sh`가 자동 생성).

---

## 1. 문제 정의

**한 줄 요약.** 사람이 컵, 책, 주전자를 다루는 동작(ParaHome)을 시뮬레이터 속 휴머노이드가 물리적으로 따라 하게 학습하고,
**물체를 쥐고 들어 올리는 데 성공하는지**, 실패한다면 **어느 순간에** 실패하는지 잰다.

| | 내용 |
|---|---|
| 입력 | 참조 시퀀스 1개 (몸 자세, 손가락 관절, 물체 궤적), 휴머노이드 MJCF (ParaHome 골격, 상자 모양 손가락), 물체/가구 모델, Isaac Gym |
| 출력 | clip 하나에 PPO 정책 하나, 그리고 과제 성공률, 들어올림 성공률, 물체 위치 오차, 넘어짐, 실패 frame, 학습 시간과 VRAM |
| 핵심 가정 | 사람 동작 데이터는 잡음이 있고 드물다 (clip 하나). 그래도 참조 주변에는 물리적으로 가능한 궤적이 많다. 그 주변 전체에서 시작해 참조로 돌아오는 법을 배우면 잡기가 강건해진다 |
| baseline 단계에서 바꾸는 것 | 없음. 논문 설정 그대로, 학습량만 하드웨어에 맞춰 줄인다 (5절) |

**왜 어려운가.** 손가락과 물체 사이 2 cm 오차만으로도 잡기가 깨진다. 모션 캡처 데이터에는 이 정도 오차가 흔하고, 그러면 참조를 그대로
따라가는 정책은 "쥐고 드는" 한 지점에서 연쇄가 끊긴다 (논문의 "chain break"). T09 WristMimic baseline이 정확히 이 지점(주전자 들기
시작)에서 0%였다. SkillMimic(v1)도 논문에서 세 clip 모두 0%다.

## 2. 왜 SkillMimic-V2인가

| 기준 | WristMimic (이전 baseline) | SkillMimic-V2 (새 baseline) |
|---|---|---|
| T09와의 관계 | 과제 안내서의 기본 baseline | 안내서에 WristMimic의 비교 대상으로 나옴. 같은 데이터(ParaHome), 같은 시뮬레이터 |
| 이 GPU에서 결과 | 6000 epoch, 성공 0%. 주전자를 쥐고 드는 순간(frame 45~60)에서 막힘 | 아직 없음 (이 제안서의 baseline) |
| 논문 성공률 (세 clip) | 주전자 장면만 시도 | Drink Cup 100%, Place Book 100%, Place Kettle 100% (표 2) |
| 2048 env (논문 설정) | 메모리 부족 → 1024로 줄임 | 들어감 (9.5~10.8 GB) |
| 학습 종료 조건 | 물체, 손목, 접촉이 벗어나면 끝 (엄격) | 넘어질 때만 끝. 랜덤 frame에서 시작 |
| 공개 코드 | 있음 | 있음. 단 ParaHome 학습 명령, 모델, history encoder는 미공개 → 코드를 읽고 맞춤 (7절) |

**주전자 clip이 다리 역할을 한다.** T09에서 실패한 물체와 같은 물체라서, "방법을 바꾸면 주전자도 되는가"를 직접 본다.
컵(한 손, 들어서 입으로)과 책(두 손으로 들어 옮기기)은 같은 방법이 다른 물체에서도 되는지를 본다.

## 3. 방법 요약 (논문 4절)

SkillMimic-V2는 **RLID**(Reinforcement Learning from Interaction Demonstration) 위에 데이터 증강 2개와 학습 기법 2개를 얹는다.

| 구성 요소 | 논문 | 이 코드에서 | ParaHome에서 |
|---|---|---|---|
| 보상 | 몸 × 물체 × 상대 위치 × 접촉, 곱 형태라 [0, 1] (식 2) | `compute_humanoid_reward`. 가중치 p 20, r 20, op 1, rel 20 (표 10) | 접촉 항은 끔 (r_cg = 1) |
| **STF** (State Transition Field) | 참조 상태 주변 ε 안에서 시작 상태를 뽑고(ε-NSI), 가장 비슷한 참조 frame으로 연결. 거리가 멀면 그 사이에 **빈(masked) frame**을 넣어 보상 없이 건너게 함 (식 6, 7) | `--state_init_random_prob 0.1` (p_n = 0.1), `noisy_resample_time`, `--enable_buffernode` (빈 frame 수 = min(−⌊log10 유사도⌋, 10)) | 사용 |
| **ATS** (Adaptive Trajectory Sampling) | 시작 frame을 그 frame에서 시작했을 때의 평균 보상이 낮을수록 자주 뽑음 (식 8) | `--reweight --reweight_alpha 1.0`. 가중치 exp(−5·r̄) | 사용 |
| **HE** (History Encoder) | 지난 60 frame을 3차원으로 압축해 정책 입력에 붙임. 미리 학습하고 고정 (식 9, 10) | `--hist_length 60 --history_embedding_size 3` | 사용 (encoder는 우리가 학습, 7절) |
| STG (Stitched Trajectory Graph) | 다른 skill의 상태에서 이 skill로 넘어가는 연결을 만듦 | `--graph_file` | **안 씀** (논문 5.3절: 물체가 서로 달라 연결이 무의미) |

- 정책: MLP 1024-512-512, 행동 분산 고정 0.055 (σ = e^−2.9), PPO (표 8). 학습 episode 60 frame.
- 논문과 코드가 다른 점: ATS의 λ_s가 논문 표 9에는 10, 코드에는 5. **코드 값을 쓴다** (논문 결과도 이 코드에서 나왔다고 본다).

## 4. 데이터: 세 clip

| clip | 내용 | 길이 | 물체 / 가구 | ParaHome 장면 | 논문 SR / ε-NSR / NR (SM + Ours) | 논문 SkillMimic(v1) SR |
|---|---|---|---|---|---|---|
| `drink_cup` | 식탁의 컵을 한 손으로 들어 입으로 가져가 마시기 | 180 frame (6초) | cup / diningtable | s6 | 100 / 33.9 / 0.89 | 0 |
| `place_book` | 책상의 책을 들어 옮겨 놓기 | 150 frame (5초) | book / desk | s6 | 100 / 82.4 / 0.86 | 0 |
| `place_kettle` | 주전자를 들어 옮겨 놓기 | 100 frame (3.3초) | kettle / diningtable | s10 | 100 / 49.9 / 0.52 | 0 |

clip은 저장소에 들어 있는 것을 그대로 쓴다. 가구 위치는 원본 ParaHome 시퀀스 s6, s10, s22에서 읽는다 (`repos/skillmimic-v2/experiments/data/parahome_seq/`, `setup.sh`가 준비).

## 5. 실험 설정 (고정)

| 항목 | 값 | 근거 |
|---|---|---|
| 방법 | SkillMimic-V2 = SM + STF + ATS + HE + buffer node (STG 없음) | 논문 5.3절의 ParaHome 설정 |
| 환경 설정 | 업스트림 `parahome_<clip>_hist60_noisyinit_simpara.yaml` 그대로 (세 clip 같은 물리, 같은 보상) | |
| env 수 | **2048** (논문과 같음) | 12GB에 들어감 (실측 9.5~10.8 GB) |
| PPO | 논문 표 8 그대로: 65,536 샘플/update, minibatch 16,384, lr 2e-5, γ 0.99, GAE 0.95, clip 0.2 | |
| 예산 | clip당 **3000 epoch = 1.97억 샘플** (논문 약 1.0B의 20%) | T09 WristMimic baseline과 **같은 샘플 수**. 논문 예산은 이 GPU에서 clip당 약 45시간 |
| seed | 0 | 시간 제약. 결과가 판정 경계(0.5 근처)면 seed 추가 |
| 체크포인트 | **10 epoch마다, 하나도 지우지 않음** (clip당 약 300개, 13 GB) | 학습이 끊겨도 잃는 것은 최대 10 epoch(약 2분) |
| 이어 학습 | 같은 명령을 다시 실행하면 마지막 체크포인트부터 epoch, optimizer, 정규화 통계, ATS 상태, 난수 상태까지 복원 | 원본 코드는 가중치만 복원했다 (7절) |

학습 시간 (실측 s/epoch 기준): 컵 약 8.9시간, 책 약 8.8시간, 주전자 약 10시간, 합계 약 28시간. 평가는 run당 약 10분.

## 6. 평가 규칙과 지표 (결과를 보기 전에 고정)

**평가 규칙.**

1. 참조 frame 2에서 시작해 300 step(10초) 진행. **처음 넘어지면 그 env는 거기서 끝** (재시작 없음. 업스트림 player는 재시작한 뒤의 frame까지 합산한다).
2. `det` = 결정적 행동 8 env. GPU PhysX가 비트 단위로 재현되지 않아 결정적이어도 여러 번 돌린다 (T09에서 확인). 8 env의 해상도는 0.125이므로 그보다 작은 차이는 "차이 없음".
3. `stoch` = 학습 때의 행동 잡음으로 32 env. `stoch_perturb` = 물체 시작 자세 교란(z축 ±45°, xy 10 cm 이내) 32 env = 논문 ε-NSR과 같은 범위.
4. 250 epoch마다의 체크포인트를 `det`로 평가해 학습 곡선을 만든다. 마지막(3000 epoch)은 세 방식 모두.
5. 학습 clip = 평가 clip. 일반화나 실제 로봇 성공을 주장하지 않는다.

**지표.**

| 지표 | 정의 | 왜 |
|---|---|---|
| **과제 성공** | clip의 모든 frame에서 물체 위치 오차 ≤ 10 cm **그리고** 넘어지지 않음 | T09의 "과제 성공"과 같은 정의 → WristMimic 결과와 바로 비교 |
| **들어올림 성공** | 참조 들어올림 높이의 50% 이상 들었고, 그때 물체가 손목 0.2 m 안 | T09가 막힌 "쥐고 들기"만 따로 본다 |
| 논문 지표 | 업스트림 metric (Place: 골반 z > 0.5, 손목-물체 < 0.2 m, 물체 z > 0.9가 60 frame 넘게. Drink: 물체 z > 1.2가 30 frame 넘게) | 논문 SR과 비교 (참고용. 주전자 참조 자체가 60 frame 조건을 45 frame만 만족) |
| 실패 frame | 처음으로 물체 오차 > 10 cm 또는 넘어진 frame | 실패가 들기 전/중/후 어디서 나는지. 참조의 들기 시작 frame과 같이 보고 |
| 그 밖 | 과제 진행률, 물체/몸 추적 오차, 넘어짐 비율, 최대 들어올림 높이, 들고 있던 frame 수, 학습 속도, VRAM | |

## 7. 원본 대비 바뀐 것 (발표 때 반드시 명시)

| 항목 | 원본 | 우리 | 동작 영향 |
|---|---|---|---|
| GPU / 학습량 | RTX 4090 24GB, clip당 약 1.0B 샘플 | RTX 5070 12GB, 1.97억 샘플 (20%) | **있음.** "논문 재현"이 아니라 "축소 예산 baseline"이라고 부른다 |
| PyTorch | conda torch | 직접 빌드한 torch 2.4.1 cu128 sm_120 (T09와 같은 휠) | 없음 |
| ParaHome 실행 경로 | README에 없음, 모델/encoder 미공개 | 코드에서 방법 ↔ task/asset/플래그 대응을 찾음 (`experiments/docs/CODE_NOTES.md` 2절). History encoder는 ParaHome 8개 clip으로 직접 학습 (3000 epoch, 6분, MSE 0.00023) | encoder가 원본과 다를 수 있음 |
| 장면 pickle 읽기 | env마다 40 MB 파일을 다시 읽음 | 경로별 1회 캐시 | 없음 (시작이 약 105초 빨라짐) |
| **이어 학습** | `--resume_from`이 가중치와 입력 정규화만 복원 → epoch 0부터 다시 세고, optimizer와 value 정규화가 초기화되고, 앞 구간의 번호 체크포인트를 **덮어씀** (점검에서 확인) | epoch, optimizer, value 정규화, frame, ATS 가중치, 난수 상태까지 저장/복원. 저장은 임시 파일에 쓰고 이름 바꾸기 (끊겨도 깨진 파일 없음) | 없음 (끊기지 않은 run과 같은 학습이 이어짐) |
| 체크포인트 주기 | 50 epoch (최신 1개만 유지) | 10 epoch, 번호 붙여 모두 유지 | 없음 |
| 평가 | player의 성공률 (재시작 frame 합산) | 우리 평가기 (`experiments/tools/evaluate.py`, 6절 규칙) | 측정 방식만 다름 |

## 8. 예상 결과와 판정 기준 (실험 전에 고정)

**예상** (`BASELINE_T09_VARIETY.md` 4절의 추정을 그대로 옮김, 확률은 대략):

| clip | 예상 | 될 확률 | 근거 |
|---|---|---|---|
| 컵 | 들어올림은 됨. 10 cm 안으로 끝까지 따라가기는 더 드묾 | 약 60% | 느슨한 종료 조건. 학습량은 논문의 20% |
| 책 | 잡고 듦, 옮기다 어긋남 | 약 50% | 논문에서 가장 강건했던 clip (ε-NSR 82.4%) |
| 주전자 | 셋 중 가장 어려움 | 약 35% | 논문에서도 SM + Ours만 성공. NR 0.52로 가장 낮음 |

**판정** (`repos/skillmimic-v2/experiments/docs/PROTOCOL.md` 5절. `write_results.sh`가 이 규칙을 그대로 적용해 적는다):

| 결과 (최종 체크포인트, det) | 해석 | T09 다음 단계 |
|---|---|---|
| clip 하나 이상에서 과제 성공 또는 들어올림 성공 ≥ 0.5 | 12GB/5070은 이 과제에 근본 장애가 아니다. WristMimic 0%의 주원인은 방법(엄격한 종료, 항상 frame 0 시작) | SkillMimic-V2 위에서 손목 항목 연구 (9절) |
| 세 clip 모두 < 0.5, 마지막 1000 epoch 동안 보상 +1% 이상 또는 과제 진행률 +0.05 이상 | 예산 부족. 메모리 문제는 아니다 (2048 env가 들어감) | 같은 run을 이어서 더 학습 (`EPOCHS=6000 bash run_cup.sh`, 체크포인트가 그대로 이어짐) |
| 세 clip 모두 < 0.5, 곡선 평평 | 이 GPU 설정에서 이 종류의 과제가 막힌다 | 물리 버퍼 점검, 과제 범위 재검토 |

"보상 +1% / 진행률 +0.05"는 PROTOCOL의 "곡선이 계속 오름"을 숫자로 정한 것이다 (2026-10-10, 결과 전).

## 9. Baseline 이후: T09 연구 질문과의 연결 (초안)

T09는 "손목 항목만 바꾸고 잡기 성공과 전신 안정성을 잰다"가 범위다. SkillMimic-V2에는 손목 전용 항목이 없다.
손목은 관절 회전 보상(모든 관절 평균)과 상대 위치 보상에 섞여 있고, 위치 보상의 key body(머리, 무릎, 팔꿈치, 발목, 손가락 끝)에는 손목이 없다.
그래서 baseline 결과를 본 뒤 **WristMimic의 손목 항목을 SkillMimic-V2에 하나씩 넣는 것**을 실험 변수로 삼는다.

| 변수 | 내용 | 기본값 (baseline) |
|---|---|---|
| 손목 reset window | 접촉 구간(잡기~들기)에서 손목이 참조에서 임계값(위치, 회전)보다 벗어나면 episode 종료. WristMimic 값 7 cm / 0.2 rad에서 시작 | 없음 (넘어질 때만 종료) |
| 손목 보상 가중치 | 손목 위치/회전 추적 항을 따로 두고 가중치를 바꿈 | 없음 (관절 회전 평균에 섞임) |
| 접촉 window 시점 | 위 window의 시작/끝 frame을 앞뒤로 옮김 | - |

- 안내서의 최소 비교와의 대응: **손목 제약 없음** = SkillMimic-V2 기본 (이 baseline), **WristMimic 기본 손목 설정** = 위 window를 WristMimic 값으로, **제안 손목 설정** = 결과를 보고 정함.
- 가설 후보 H1: 주전자 clip에 WristMimic의 손목 reset window를 넣으면 들어올림 성공률이 baseline보다 0.25 이상 떨어진다
  (= 엄격한 종료 조건이 T09 0%의 원인이라는 설명의 직접 검증). 차이가 0.25 미만이면 기각.
- 가설 후보 H2: 손목 보상 가중치를 높이면 손목 오차는 줄지만 넘어짐과 몸통 기울기가 늘어난다. 안정성 지표가 나빠지지 않으면 기각.
- 필요한 추가 작업: 평가기에 손목 추적 오차, 발 미끄러짐, 몸통 기울기, 관절 한계 도달 비율을 기록으로만 추가 (동작 불변).
- 이 대응은 **baseline 결과와 담당 교수 확인 뒤에 확정**한다. 지금 확정하지 않는다.

## 10. 실행 계획 (사용자가 터미널에서 직접, GPU 작업은 한 번에 하나)

```bash
cd ~/code/3dv-project/repos/skillmimic-v2
bash run_cup.sh       # 약 9시간 + 평가 10분
bash run_book.sh      # 약 9시간 + 평가 10분
bash run_kettle.sh    # 약 10시간 + 평가 10분
bash write_results.sh # -> ~/code/3dv-project/BASELINE_SKILLMIMIC.md (GPU 안 씀, 언제든 다시 실행 가능)
```

| 순서 | 할 일 | 시간 | 확인할 것 |
|---|---|---|---|
| 0 | 파이프라인 점검 (끝남, 2026-10-10) | 15분 | 끊고 다시 실행했을 때 epoch, optimizer, ATS 상태가 이어지는가 |
| 1 | 컵 | 약 9시간 | 1000 epoch(약 3시간)에서 TensorBoard 보상 곡선. 평평하면 8절 판정표 3행을 의심 |
| 2 | 책 | 약 9시간 | |
| 3 | 주전자 | 약 10시간 | T09와의 직접 비교 |
| 4 | `write_results.sh`, 실패 영상 녹화 (viewer, env 1개) | 수 분 | 판정, 실패 frame vs 참조 들기 시작 frame |
| 5 | 9절 변수 확정, 교수 확인 | - | |

- **끊기면 같은 명령을 다시 실행한다.** 학습은 마지막 체크포인트(최대 10 epoch 전)부터, 평가는 남은 것만 한다.
- 중간에 체크포인트를 평가하고 싶으면 Ctrl-C로 멈추고 `bash experiments/scripts/eval_run.sh <run> curve 1000` 후 다시 `run_cup.sh`. 학습은 그대로 이어진다.
- 디스크: 체크포인트 run당 약 13 GB, 세 run 약 39 GB (여유 818 GB).

## 11. 위험과 대응

| 위험 | 대응 |
|---|---|
| baseline 변경을 과제 범위 밖으로 볼 수 있음 | SkillMimic-V2는 안내서에서 WristMimic의 비교 대상. 손목 항목을 실험 변수로 유지 (9절). 첫 결과와 함께 교수 확인 |
| 3000 epoch로는 부족 (논문의 20%) | 판정표 2행. 체크포인트가 완전히 이어지므로 같은 run을 늘려 학습 |
| 2048 env에서 메모리 부족 | 실측 최대 10.8 GB (여유 1~2.5 GB). 학습 중 GPU를 쓰는 다른 프로그램을 끈다. 그래도 안 되면 `NUM_ENVS=1536` (보고서에 명시) |
| 결과가 0.5 근처에서 애매함 | seed 1, 2 추가. 8 env 해상도(0.125) 안의 차이는 "차이 없음" |
| History encoder가 원본과 다름 | 학습 설정을 업스트림 스크립트와 같게 했고 로그를 남김 (`experiments/logs/hist_encoder.log`). 결과 해석 때 명시 |
| GPU 비결정성 | 결정적 평가도 8 env 반복 (6절) |

## 12. 참고

- SkillMimic-V2: Learning Robust and Generalizable Interaction Skills from Sparse and Noisy Demonstrations, Yu et al., SIGGRAPH 2025 (arXiv 2505.02094)
- SkillMimic: Learning Basketball Interaction Skills from Demonstrations, Wang et al., CVPR 2025 (arXiv 2408.15270)
- WristMimic: Full-Body Humanoid Control with Wrist-Guided Manipulation, Yu et al., ECCV 2026 (arXiv 2607.06438)
- ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions, Kim et al., 2024
- 이 프로젝트: `BASELINE_T09.md` (WristMimic 0%), `BASELINE_T09_VARIETY.md` (이번 선택의 배경), `repos/skillmimic-v2/experiments/docs/{PROTOCOL,CODE_NOTES}.md`
