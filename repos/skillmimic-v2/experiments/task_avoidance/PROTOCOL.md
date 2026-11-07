# 분석 규칙과 해석 범위

프로토콜 파일: `experiments/configs/task_avoidance/protocol.json`. 실행 전 내용을 고정하고 해시와 복사본을 저장한다. 필수 검사 실패를 성공시키기 위해 결과 확인 후 임계값을 바꾸지 않는다. 변경 시 새 protocol/version 연구 기록과 새 label이 필요하다.

## 가설과 반증

H1: 같은 초기 상태에서 후기 정책은 조작 시도·목표 이동이 감소한다.
H2: 그 조건에서 기존 fix의 유한 구간 보상 합계는 증가한다.
H3: 같은 정책에서 출발한 보상 개입이 조작 감소를 완화한다.

H1만이면 행동 감소, H1+H2면 보상/조작 불일치, H1+H2+H3이면 보상 유인 관여를 지지한다. 어느 것도 의도·전역 최적 전략·일반적인 과제 회피를 증명하지 않는다. 조작과 보상 모두 감소하면 능력 저하·학습 불안정 가설도 검토한다. 변화가 작거나 seed마다 다르면 INCONCLUSIVE로 보고한다. 최소 효과 크기와 확증용 seed 수는 pilot 변동을 바탕으로 별도 확증 실험 전에 정해야 한다. 현재 코드는 유의성/회피 PASS 판정을 자동 생성하지 않는다.

## 초기 상태와 실행 통제

- task/asset/data/history/phase/fix 설정 유지. 행동은 deterministic이며 초기화는 기존 YAML의 noisy init0.1을 보존.
- Python/numpy/torch/CUDA RNG를 정책 로드 후 모두 재설정. fresh process로 simulator와 task 이력의 checkpoint 간 오염을 피함.
- 실제 actor root/DOF/rigid body/contact/물체/history/이전·현재 HOI/관측/정책 raw input/phase/buffer를 NPZ로 저장. 같은 evaluation seed와 env ID끼리 비교. 기본 initial atol=0, rtol=0; 일치하지 않으면 중단.
- 이 snapshot은 비교 증거이며 simulator contact solver cache를 완전히 직렬화한 state bank는 아니다. fresh construction+repeat probe로 보완하되 완전한 물리 결정성은 주장하지 않음.
- 고정 frame2에서 초기화해도 기존 STF/noisy init은 참조 frame을 다시 선택할 수 있다. 실제 start_frames를 저장·pair하고 각 episode의 유효 참조 길이에서 집계를 끝낸다. 명목 frame2라고 모든 실제 phase가2라고 가정하지 않음.
- 32환경×3 seed=96개가 모두 서로 다른 상태라는 보장은 없음. 중복 상태가 있을 수 있으며 표본을 독립 학습 반복으로 세지 않음.

## 보상과 종료의 정확한 범위

원 reward의 최종 식은 `rb * ro * rig`이다 (`skillmimic_parahome.py::compute_humanoid_reward`). diagnostic은 해당 함수 AST의 산술을 그대로 실행하고 마지막 return만 factor stack으로 바꾼 별도 사본이다. 원 JIT 함수 반환값은 그대로 사용하며 diagnostic 재구성값과 매 step 비교한다. 항별 평균은 보상 성분 설명이며 가산 기여도/인과 효과가 아니다. 실제 buffer 처리 후 delivered reward와도 비교하여 다르면 중단한다.

기존 fix `--test`는 학습의 물체 오차 종료를 비활성화한다 (`skillmimic_parahome_localhist.py::_compute_reset`). 따라서 두 return을 분리한다.

1. return_eval: 해당 episode의 유효 참조 구간, 실제 done/넘어짐까지 기존 reward의 γ=0.99 할인 합.
2. return_train_window: 같은 궤적에서 기존 학습 물체 종료 predicate 및 min(60, 참조 잔여길이) 시간 제한을 적용한 관측 prefix 할인 합. terminal reward는 포함하고 그 뒤는0. 물체 predicate는 실제 학습과 같은 `_curr_obs` 및 `_curr_ref_obs`를 사용.

이 prefix는 **실제 학습 분포에서 기대한 수익이나 PPO 목적함수가 아니다.** ATS 초기화 분포, buffer, critic timeout bootstrap 등을 재현하지 않는다. 최대60step observed return의 유인 진단이다. 실제 timeout 시 critic은 bootstrap하지만 본 분석은 관측 reward만 비교한다. 이 한계 때문에 H2 단독으로 학습의 원인을 확정하지 않는다.

현재 post_physics_step은 reset 판정 후 progress counter를 증가시킨다. worker는 판정시 counter=반환된 progress-1을 사용하여 off-by-one을 피한다. 첫 post-step reward는 upstream의 참조/현재 HOI 시간 경로 그대로이며 해당 경로의 정확성 V&V를 이번 작업에서 주장하지 않는다.

## 지표

- lift_attempt: 실제 초기 물체 z에서5cm 이상 상승하고 wrist-object<20cm, nonfall 유효 transition이 존재. '의도'가 아니라 관측 proxy. 손목 근접은 접촉 아님.
- lift_success: 동일 조건에서 전체 참조의 최대 높이 상승량50% 이상. 기존 legacy 지표와 달리 유효 clip 구간과 실제 초기 object 높이를 사용하므로 legacy 결과와 섞지 않음.
- goal_progress_m: 실제 initial object→마지막 유효 상태까지, 고정 최종 참조 위치에 대한 거리 감소. terminal transition 포함.
- stable_lift: 같은 episode에서 lift_success AND 유효 참조 구간 전체를 넘어짐/조기 종료 없이 완료.
- task_success: 같은 유효 참조 구간 전체에서 object tracking error≤10cm 및 넘어짐/조기 종료 없음.
- goal_error_survivors_m은 완료한 episode에만 정의. 누락값을0으로 채우지 않음.
- first_tracking_failure_step은10cm tracking 또는 유효성 기준의 첫 위반, 실패 원인 진단 아님.
- wrist_near_steps, held_steps, fall, 실제 유효 step 수, 실제 참조 시작 phase를 함께 보고.

분석 단위는 (정책, evaluation seed, env ID). paired difference는 right-left. eval seed별 평균을 먼저 보고한다. 독립 학습 seed가 하나인데96 episode를 학습 반복으로 취급한 p-value/CI는 만들지 않는다. 그래프는 evaluation seed 평균의 기술통계이며 신뢰구간이 아니다. 영상 판정은 별도로 고정 선정 규칙과 수동 기록이 필요하며 이 도구는 영상을 자동 녹화하지 않는다.

## C 실험 통제와 남는 한계

동일 source checkpoint의 optimizer/normalizer/ATS/epoch/frame을 복원하고 C0/C1에 동일 continuation RNG seed를 사용한다. 초기 전체 episode 물리 상태는 새로 reset된다. 원 source의 checkpoint/run_meta와 현재 파일 hash를 기록하지만 옛 학습 당시 데이터·설정 hash가 없으면 완전한 원조건 일치를 입증할 수 없다.

ATS 알고리즘 설정은 유지하되 변경 reward로 인해 이후 sampling 분포는 달라질 수 있다. 효과는 그 상호작용을 포함한 보상 개입 전체 효과다. 원 source에서 복원한 value head의 목적함수도 C1에서 바뀌므로 적응 비용이 섞인다. C2 단순 위치 보상 강화 대조군은 후속 계획이며 아직 구현하지 않았다. 독립 from-scratch A/B는 기존 progress 패키지에서 별도 실행한다.

분기 반복 결과를 보고 가중치/source를 선택하면 선택 실험으로 기록한다. 최종 확증은 새 독립 source seeds와 사전에 고정한 평가 집합에서 시행한다. 이미 본 1000/3000 및 evaluation seeds의 결과를 같은 의미의 held-out 결과로 포장하지 않는다.
